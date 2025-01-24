import os
import ray
import boto3
import tempfile
import torch
import argparse
from functools import partial
import deepchem as dc
from data_utils import RayDataset
from deepchem.models.optimizers import Lamb
from ray import train
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig, FailureConfig, SyncConfig


def restore_v2(  # type: ignore
        self, checkpoint=None, model_dir=None) -> None:  # type: ignore
    """Reload the values of all variables from a checkpoint file.

    Parameters
    ----------
    checkpoint: str
        the path to the checkpoint file to load.  If this is None, the most recent
        checkpoint will be chosen automatically.  Call get_checkpoints() to get a
        list of all available checkpoints.
    model_dir: str, default None
        Directory to restore checkpoint from. If None, use self.model_dir.  If
        checkpoint is not None, this is ignored.
    """
    # FIXME I am rewriting restore because the restore method in parent class
    # does not restore layers which are not components. This restore method
    # can restore an full model.

    self._ensure_built()
    if checkpoint is None:
        checkpoints = sorted(self.get_checkpoints(model_dir))
        if len(checkpoints) == 0:
            raise ValueError('No checkpoint found')
        checkpoint = checkpoints[0]
    data = torch.load(checkpoint, map_location=self.device)
    self.model.load_state_dict(data['model_state_dict'], strict=False)
    if 'optimizer_state_dict' in data:
        print('restoring checkpoint with optimizer state dict')
        self._pytorch_optimizer.load_state_dict(data['optimizer_state_dict'])
    else:
        print('restoring checkpoint without optimizer state dict')
    self._global_step = data['global_step']


LR_SCHEDULE_MAPPINGS = {
    'polynomial_decay': {
        "scheduler_class": dc.models.optimizers.PolynomialDecay,
        "init_params": {
            "required_init_params":
            ["initial_rate", "final_rate", "decay_steps"],
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
            "required":
            ["initial_rate", "num_warmup_steps", "num_training_steps"],
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
        if world_rank == 0 and (step % model.checkpoint_frequency == 0
                                or step == model.total_steps):
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
    print("torch version ------------------------> ", torch.__version__)
    default_dtype = torch.get_default_dtype()
    print("torch default dtype ------------------> ", default_dtype)
    device = ray.train.torch.get_device()
    model_name = config['model_name']
    """ The `"learning_rate"` parameter in `init_kwargs` can either be the name of a Learning Rate Scheduler or a float value.
    The available LR Scheduler types are `'polynomial_decay'`, `'exponential_decay'`, `'linear_cosine_decay'`, and `'piecewise_constant_schedule'`.
    If `"learning_rate"` is a string, the system checks whether it corresponds to a valid scheduler type in the 
    mapping and verifies that all required parameters for the scheduler class are present in `init_kwargs`.
    It then initializes the scheduler and passes it as the "learning_rate" parameter to the model."""

    init_kwargs = config['init_kwargs']
    if 'learning_rate' in init_kwargs.keys():
        if isinstance(init_kwargs['learning_rate'], str):
            lr_scheduler = init_kwargs['learning_rate']
            if lr_scheduler not in LR_SCHEDULE_MAPPINGS:
                raise ValueError(
                    "Learning rate scheduler type not recognized.")
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
        init_kwargs['optimizer'] = Lamb(learning_rate=learning_rate_scheduler)

    total_steps = init_kwargs['total_steps']
    del init_kwargs['total_steps']
    if 'checkpoint_frequency' in init_kwargs:
        checkpoint_frequency = init_kwargs['checkpoint_frequency']
        del init_kwargs['checkpoint_frequency']
    else:
        checkpoint_frequency = 100

    dc_model = MODEL_MAPPINGS[model_name](**init_kwargs)
    print('model initialized successfully')
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
            model_dir = os.path.join(ckpt_dir, 'model_ckpt')
            data_path = os.path.join(ckpt_dir, 'data.pt')
            restore_v2(dc_model, model_dir=model_dir)
            data = torch.load(data_path)
            start_epoch = data['epoch']
    else:
        print('checkpoint is not present')
        start_epoch = 0

    train_dataset = RayDataset(train.get_dataset_shard("train"))

    for epoch in range(start_epoch, config['num_epochs']):
        callback = partial(_callback, epoch=epoch)
        dc_model.fit(train_dataset, nb_epoch=1, callbacks=[callback])


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_path', type=str)
    args = argparser.parse_args()

    s3_client = boto3.client("s3")
    bucket_name = args.config_path.split('/')[0]
    config = ray.data.read_json(args.config_path)

    new_config = {}
    new_config['dataset_path'] = config.to_pandas()['dataset_path'].tolist()[0]
    new_config['exp_name'] = config.to_pandas()['exp_name'].tolist()[0]
    new_config['storage_path'] = config.to_pandas()['storage_path'].tolist()[0]
    new_config['num_epochs'] = config.to_pandas()['num_epochs'].tolist()[0]
    new_config['init_kwargs'] = config.to_pandas()['init_kwargs'].tolist()[0]
    new_config['model_name'] = config.to_pandas()['model_name'].tolist()[0]
    new_config['num_workers'] = config.to_pandas()['num_workers'].tolist()[0]
    new_config['checkpoint_path'] = config.to_pandas(
    )['checkpoint_path'].tolist()[0]
    checkpoint_path = new_config['checkpoint_path']
    # TODO Add an user argument in frontend to speciyfy experiment name.

    # ray.data.read_binary_files
    dataset = RayDataset.read(new_config['dataset_path'])
    ckpt_config = CheckpointConfig(num_to_keep=5,
                                   checkpoint_score_attribute=LOSS_ARG_NAME,
                                   checkpoint_score_order='min')
    failure_config = FailureConfig(max_failures=200)
    sync_config = SyncConfig(sync_period=120,
                             sync_artifacts=False,
                             sync_artifacts_on_checkpoint=True)
    run_config = RunConfig(checkpoint_config=ckpt_config,
                           failure_config=failure_config,
                           sync_config=sync_config,
                           name=new_config['exp_name'],
                           storage_path=new_config['storage_path'])

    datasets = {"train": dataset.dataset}
    dataset_length = dataset.dataset.count()
    batch_size = new_config['init_kwargs']['batch_size']
    num_epochs = new_config['num_epochs']
    num_workers = new_config['num_workers']
    total_steps = int(
        ((dataset_length / batch_size) / num_workers) * num_epochs)
    new_config['init_kwargs']['total_steps'] = total_steps

    train_loop_config = {
        'num_epochs': new_config['num_epochs'],
        'batch_size': new_config['init_kwargs']['batch_size'],
        'model_name': new_config['model_name'],
        'init_kwargs': new_config['init_kwargs']
    }
    scaling_config = ScalingConfig(num_workers=new_config['num_workers'],
                                   use_gpu=True)

    exp_path = os.path.join(new_config['storage_path'], new_config['exp_name'])

    if TorchTrainer.can_restore(exp_path):
        # By default, experiments are restored if a checkpoint exists.
        # Otherwise, a new experiment is started unless force_restore is False.
        print("experiment can be restored")
        trainer = TorchTrainer.restore(exp_path, datasets=datasets)
    else:
        if checkpoint_path is not None:
            print("experiment can be restored from the given checkpoint:",
                  checkpoint_path)
            ckpt = Checkpoint(checkpoint_path)
            trainer = TorchTrainer(
                train_loop_per_worker=_train_loop_per_worker,
                train_loop_config=train_loop_config,
                datasets=datasets,
                scaling_config=scaling_config,
                run_config=run_config,
                resume_from_checkpoint=ckpt)
        else:
            print("experiment cannot be restored")
            trainer = TorchTrainer(
                train_loop_per_worker=_train_loop_per_worker,
                train_loop_config=train_loop_config,
                datasets=datasets,
                scaling_config=scaling_config,
                run_config=run_config)
    result = trainer.fit()
    ckpt = result.get_best_checkpoint(metric='iteration_loss', mode='min')
    ckpt_path = ckpt.path
    print(
        "Training completed successfully. Checkpoint path of the trained model:",
        ckpt_path)
