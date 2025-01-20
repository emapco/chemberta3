import torch
import yaml
import tempfile
import os
import deepchem as dc
import ray
import numpy as np
from ray import train
from data_utils import RayDataset
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig, FailureConfig
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.data.data_collator import DataCollatorForLanguageModeling


LR_SCHEDULE_MAPPINGS = {
    'polynomial_decay': {
        "scheduler_class": dc.models.optimizers.PolynomialDecay,
        "init_params": {
            "required_init_params": [
                "initial_rate", "final_rate", "decay_steps"
            ],
            "optional_init_params": ["power"]
        }
    },
    'exponential_decay': {
        "scheduler_class": dc.models.optimizers.ExponentialDecay,
        "init_params": {
            "required": ["initial_rate", "decay_rate", "decay_steps"],
            "optional": ["staircase"]
        }
    },
    'linear_cosine_decay': {
        "scheduler_class": dc.models.optimizers.LinearCosineDecay,
        "init_params": {
            "required": ["initial_rate", "decay_steps"],
            "optional": ["alpha", "beta", "num_periods"]
        }
    },
    'piecewise_constant_schedule': {
        "scheduler_class": dc.models.optimizers.PiecewiseConstantSchedule,
        "init_params": {
            "required": ["initial_rate"],
            "optional": ["boundaries_and_scales"]
        }
    },
    'lambda_lr_with_warmup': {
        "scheduler_class": dc.models.optimizers.LambdaLRWithWarmup,
        "init_params": {
            "required": [
                "initial_rate", "num_warmup_steps", "num_training_steps"
            ],
            "optional": ["warmup_type"]
        }
    },
}


MODEL_MAPPINGS = {}
MODEL_MAPPINGS['chemberta'] = dc.models.torch_models.chemberta.Chemberta
MODEL_MAPPINGS['grover'] = dc.models.torch_models.GroverModel
MODEL_MAPPINGS['molformer'] = dc.models.torch_models.MoLFormer

LOSS_ARG_NAME = 'iteration_loss'


def _callback(model, step, iteration_loss, epoch):
    """Callback to pass during model.fit call.

    Parameters
    ----------
    model: dc.models.Model
        A DeepChem model
    loss: float
        Loss in an iteration
    step: int
        Training iteration

    NOTE: Callback is important as the ray.train.report method ensures synchronization
    of gradients across workers. Otherwise, training will hang.
    """
    world_rank = ray.train.get_context().get_world_rank()
    loss = iteration_loss.detach().cpu().numpy().item()
    metrics = {'iteration': step, LOSS_ARG_NAME: loss}
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        checkpoint = None
        if world_rank == 0 and (step % model.checkpoint_frequency == 0 or
                                step == model.total_steps):
            model_dir = os.path.join(temp_checkpoint_dir, 'model_ckpt')
            data_path = os.path.join(temp_checkpoint_dir, 'data.pt')
            os.makedirs(model_dir)
            data = {'epoch': epoch}

            model.save_checkpoint(model_dir=model_dir)
            torch.save(data, data_path)

            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        ray.train.report(metrics, checkpoint=checkpoint)


def _train_loop_per_worker(config):
    """Ray training loop"""
    device = ray.train.torch.get_device()
    model_name = config['model_name']

    # The `"learning_rate"` parameter in `init_kwargs` can either be the name of a Learning Rate Scheduler or a float value.
    # The available LR Scheduler types are `'polynomial_decay'`, `'exponential_decay'`, `'linear_cosine_decay'`, and `'piecewise_constant_schedule'`.
    # If `"learning_rate"` is a string, the system checks whether it corresponds to a valid scheduler type in the mapping and verifies that all required parameters for the scheduler class are present in `init_kwargs`.
    # It then initializes the scheduler and passes it as the "learning_rate" parameter to the model.
    init_kwargs = config['init_kwargs']
    if 'learning_rate' in init_kwargs.keys():
        if isinstance(init_kwargs['learning_rate'], str):
            lr_scheduler = init_kwargs['learning_rate']
            if lr_scheduler not in LR_SCHEDULE_MAPPINGS:
                raise ValueError("Learning rate scheduler type not recognized.")
            lr_scheduler_params = {}
            lr_scheduler_params_dict = LR_SCHEDULE_MAPPINGS[lr_scheduler][
                'init_params']
            for type, args in lr_scheduler_params_dict.items():
                for arg in args:
                    if str(arg) in init_kwargs:
                        lr_scheduler_params[arg] = init_kwargs[arg]
                    else:
                        if type == 'required':
                            raise Exception(
                                f"Missing required parameters for Learning Rate Scheduler: {lr_scheduler}"
                            )
            print(lr_scheduler_params)
            learning_rate_scheduler = LR_SCHEDULE_MAPPINGS[lr_scheduler][
                'scheduler_class'](**lr_scheduler_params)

        elif isinstance(init_kwargs['learning_rate'], float):
            learning_rate_scheduler = init_kwargs['learning_rate']

        init_kwargs['learning_rate'] = learning_rate_scheduler
        init_kwargs['optimizer'] = dc.models.optimizers.Lamb(learning_rate=learning_rate_scheduler)

    total_steps = init_kwargs['total_steps']
    del init_kwargs['total_steps']
    
    # `checkpoint_frequency` determines how often checkpoints are created during training. By default it is set to 100 iterations.
    if 'checkpoint_frequency' in init_kwargs:
        checkpoint_frequency = init_kwargs['checkpoint_frequency']
        del init_kwargs['checkpoint_frequency']
    else:
        checkpoint_frequency = 100
    dc_model = MODEL_MAPPINGS[model_name](**init_kwargs)
    dc_model.checkpoint_frequency = checkpoint_frequency
    dc_model.total_steps = total_steps
    checkpoint = train.get_checkpoint()

    # model should be prepared for the distributed format before restoring the checkpoint
    # otherwise the model restore fails with error that no params found with `module.` prefix.
    dc_model.model = ray.train.torch.prepare_model(dc_model.model)
    dc_model.device = device
    if checkpoint:
        print('checkpoint is present')
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_dict = torch.load(os.path.join(ckpt_dir, 'ckpt.pt'),
                                   weights_only=True)
            # We are checkpointing only the PyTorch model and not using DeepChem checkpointing.
            model.load_state_dict(ckpt_dict['model'])
            optimizer.load_state_dict(ckpt_dict['optimizer'])
            start_epoch = int(ckpt_dict['epoch']) + 1
            loss = ckpt_dict['loss']
    else:
        print('checkpoint is not present')
        start_epoch = 0

    train_dataset = RayDataset(train.get_dataset_shard("train"))
    train_dataloader = train_data_shard.iter_batches(batch_size=config["batch_size"])

    for epoch in range(start_epoch, config['num_epochs']):
        callback = partial(_callback, epoch=epoch)
        dc_model.fit(train_dataset, nb_epoch=1, callbacks=[callback])

if __name__ == '__main__':
    with open('config.yaml', 'r'):
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_parameters = config['model_parameters']
    model_name = config['model_name']
    if config['is_csv']:
        train_dataset = ray.data.read_csv(config['dataset_path'])

    # dataset_path = train_data_dir
    # train_dataset = RayDataset.read(dataset_path).dataset
    train_loop_config = {"num_epochs": config['num_epochs'], "model_name": model_name, "init_kwargs": config['init_kwargs']}
    storage_path, exp_name = config['storage_path'], config['exp_name']

    # `num_to_keep` parameter specifies that a maximum of 5 checkpoints will be retained at any given time. Older checkpoints will be 
    # discarded as new ones are created, ensuring storage efficiency. `checkpoint_score_order`=`min` indicates that the checkpoint with 
    # the smallest loss will be preferred.
    ckpt_config = CheckpointConfig(num_to_keep=5,
                                   checkpoint_score_attribute=LOSS_ARG_NAME,
                                   checkpoint_score_order='min')

    # In a distributed system, tasks may fail intermittently due to network issues, resource constraints, or other transient errors. 
    # This setting allows the system to tolerate a specific number of such failures before considering the process unsuccessful.
    failure_config = FailureConfig(max_failures=200)                               
    run_config = RunConfig(checkpoint_config=ckpt_config,
                           failure_config=failure_config,
                           name=exp_name,
                           storage_path=storage_path)
    exp_path = os.path.join(storage_path, exp_name)
    scaling_config = ScalingConfig(num_workers=config['num_workers'],
                                   use_gpu=True)

    datasets = {"train": train_dataset}
    if TorchTrainer.can_restore(exp_path):
        # By default, experiments are restored if a checkpoint exists.
        # Otherwise, a new experiment is started unless force_restore is False.
        print("experiment can be restored")
        trainer = TorchTrainer.restore(exp_path, datasets=datasets)
        result = trainer.fit()
    else:
        print("experiment cannot be restored \n starting a new experiment")
        trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker,
                               train_loop_config=train_loop_config,
                               datasets={"train": train_dataset},
                               scaling_config=scaling_config,
                               run_config=run_config)
        result = trainer.fit()
