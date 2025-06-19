import os
import time
import uuid
import torch
import argparse
import logging
import numpy as np
import deepchem as dc
import random
import shutil
from datetime import datetime
from deepchem.models.optimizers import Optimizer, LearningRateSchedule
from apex.optimizers import FusedLAMB
from deepchem.models.torch_models import MoLFormer
from typing import List, Tuple, Callable, Union
from collections.abc import Sequence as SequenceCollection
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.utils.typing import OneOrMany
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator
from typing import Callable, List, Tuple, Union

_logger = logging.getLogger(__name__)

class CustomMoLFormer(MoLFormer):
    """
    This is a customer version of molformer which can use unique ids for `variable` params in fit method.
    It enables the use of FusedLamb optimizer with `decay` and `non-decay` param groups.
    """
    def fit_generator(self,
                      generator,
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      restore: bool = False,
                      variables = None,
                      loss = None,
                      callbacks = [],
                      all_losses = None) -> float:
        """Train this model on data from a generator with variables as dictionary of list of torch.nn.Parameter with unique UUID

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        variables: Optional[Dict]
            dictionary of list of torch.nn.Parameter with unique UUID
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step, **kwargs) that will be invoked
            after every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.

        Returns
        -------
        The average loss over the most recent checkpoint interval
        """
        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        self.model.train()
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = list(variables.keys())[0]
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables[var_key])
                if isinstance(self.optimizer.learning_rate,
                              LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                        optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
        time1 = time.time()

        # Main training loop.

        for batch in generator:
            if restore:
                self.restore()
                restore = False
            inputs: OneOrMany[torch.Tensor]
            inputs, labels, weights = self._prepare_batch(batch)

            optimizer.zero_grad()
            outputs = self.model(**inputs)

            batch_loss = outputs.get("loss")
            batch_loss.backward()
            optimizer.step()
            if lr_schedule is not None:
                lr_schedule.step()
            self._global_step += 1
            current_step = self._global_step

            avg_loss += batch_loss

            # Report progress and write checkpoints.
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                _logger.info('Ending global_step %d: Average loss %g' %
                            (current_step, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0

            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                try:
                    # NOTE In DeepChem > 2.8.0, callback signature is updated to allow
                    # variable arguments.
                    c(self, current_step, iteration_loss=batch_loss)
                except TypeError:
                    # DeepChem <= 2.8.0, the callback should have this signature.
                    c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', batch_loss,
                                                current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': batch_loss})
                self.wandb_logger.log_data(all_data, step=current_step)

        # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            _logger.info('Ending global_step %d: Average loss %g' %
                        (current_step, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss

        if checkpoint_interval > 0:
            self.save_checkpoint(max_checkpoints_to_keep)

        time2 = time.time()
        _logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss


# Set seeds
def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed: int
        Random seed to set.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(dataset: str, epochs: int,
                  batch_size: int, splits_name: str) -> logging.Logger:
    """Set up logging for the experiment.

    Parameters
    ----------
    dataset: str
        Name of the dataset being used.
    epochs: int
        Number of epochs for training.
    batch_size: int
        Batch size used for training.
    splits_name: str
        Name of the splits to use for the datasets.

    Returns
    -------
    logger : logging.Logger
        Configured logger for the experiment.
    """

    # Create a directory for logs if it doesn't exist
    log_dir = f'logs_{splits_name}_MoLFormer'
    os.makedirs(log_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        log_dir,
        f"MoLFormer_{splits_name}_run_{dataset}_epochs{epochs}_batch_size{batch_size}_{datetime_str}.log"
    )

    logger = logging.getLogger(
        f"logs_MoLFormer_{dataset}_epochs{epochs}_batch_size{batch_size}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def get_param_groups(model: torch.nn.Module):
    """
    Separates model parameters into decay and no_decay groups.

    Parameters
    ----------
    model: torch.nn.Module
        The model containing parameters.

    Returns
    -------
    optim_groups: List[Dict]
        A list of parameter groups with associated weight decay settings.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups


class FusedLamb(Optimizer):

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08):
        """Construct an Lamb optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        beta1: float
            a parameter of the Lamb algorithm
        beta2: float
            a parameter of the Lamb algorithm
        epsilon: float
            a parameter of the Lamb algorithm
        """
        super(FusedLamb, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_pytorch_optimizer(self, params):
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return FusedLAMB(params,
                             lr=lr,
                             betas=(self.beta1, self.beta2),
                             eps=self.epsilon)


def model_fn(tasks: List, model_dir: str, learning_rate: float,
             batch_size: int, pretrained_model_path: str) -> MoLFormer:
    # training model
    """Create a MoLFormer model for classification tasks.

    Parameters
    ----------
    tasks: List
        List of tasks for the model.
    model_dir: str
        Directory to save the model.
    learning_rate: float
        Learning rate for the model.
    batch_size: int
        Batch size for the model.
    pretrained_model_path: str
        Path to the pretrained MoLFormer model.

    Returns
    -------
    model: MoLFormer
        MoLFormer model for classification tasks.
    """

    finetune_model = CustomMoLFormer(task='classification',
                               optimizer=FusedLamb(learning_rate=learning_rate),
                               batch_size=batch_size,
                               n_tasks=len(tasks),
                               model_dir=model_dir)
    finetune_model.load_from_pretrained(pretrained_model_path)
    return finetune_model


def run_deepchem_experiment(run_id: int,
                            model_fn: Callable,
                            train_dataset: dc.data.DiskDataset,
                            valid_dataset: dc.data.DiskDataset,
                            test_dataset: dc.data.DiskDataset,
                            metric: dc.metrics.Metric,
                            dataset: str,
                            tasks: List,
                            model_dir: str,
                            batch_size: int,
                            learning_rate: float,
                            pretrained_model_path: str,
                            epochs: int = 100,
                            logger: logging.Logger = None) -> float:
    """Run a single experiment with the given parameters.

    Parameters
    ----------
    run_id: int
        ID of the current run.
    model_fn: function
        Function to create the model.
    train_dataset: dc.data.DiskDataset
        Training dataset.
    valid_dataset: dc.data.DiskDataset
        Validation dataset.
    test_dataset: dc.data.DiskDataset
        Test dataset.
    metric: dc.metrics.Metric
        Metric to evaluate the model.
    dataset: str
        Name of the dataset being used.
    tasks: List
        List of tasks for the model.
    model_dir: str
        Directory to save the model.
    batch_size: int
        Batch size for the model.
    learning_rate: float
        Learning rate for the model.
    pretrained_model_path: str
        Path to the pretrained CustomMoLFormer model.
    epochs: int
        Number of epochs for training.
    logger: logging.Logger
        Logger for the experiment.

    Returns
    -------
    test_score: float
        Test score of the model.
    """

    set_seed(run_id)
    model = model_fn(tasks=tasks,
                     model_dir=model_dir,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     pretrained_model_path=pretrained_model_path)
    best_score = -np.inf

    # Get current datetime and format it
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    best_model_dir = f"checkpoints_{dataset}/best_model_epochs{epochs}_run{run_id}_{current_datetime}"
    os.makedirs(f"checkpoints_{dataset}", exist_ok=True)

    # used to assign a unique id to params that remains constant through all epochs
    params_uuid = uuid.uuid4()
    for epoch in range(epochs):

        optim_param_groups = get_param_groups(model=model.model)
        loss = model.fit(dataset=train_dataset,
                         nb_epoch=1,
                         restore=epoch > 0,
                         max_checkpoints_to_keep=1,
                         variables={params_uuid: optim_param_groups})
        scores = model.evaluate(valid_dataset, [metric])
        val_score = scores[metric.name]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Val {metric.name}: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            # model.save_checkpoint(max_checkpoints_to_keep=1)

            # Save current checkpoint as best model
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
            logger.info(f"Global step of best model: {model._global_step}")
            shutil.copytree(model.model_dir, best_model_dir)

            logger.info(
                f"Best model saved at epoch {epoch+1} with val {metric.name}: {val_score:.4f}"
            )

    # Load best checkpoint before evaluating on test set
    model.restore(model_dir=best_model_dir)
    test_scores = model.evaluate(test_dataset, [metric])
    test_score = test_scores[metric.name]
    logger.info(f"Test {metric.name}: {test_score:.4f}")

    return test_score


def triplicate_benchmark_dc(
        dataset: str,
        splits_name: str,
        model_fn: Callable,
        metric: dc.metrics.Metric,
        tasks: List,
        batch_size: int,
        learning_rate: float,
        pretrained_model_path: str,
        nb_epoch: int = 100,
        logger: logging.Logger = None) -> Tuple[float, float]:
    """Run a triplicate benchmark for the given dataset.

    Parameters
    ----------
    dataset: str
        Name of the dataset being used.
    splits_name: str
        Name of the splits to use for the datasets.
    model_fn: Callable
        Function to create the model.
    metric: dc.metrics.Metric
        Metric to evaluate the model.
    tasks: List
        List of tasks for the model.
    batch_size: int
        Batch size for the model.
    learning_rate: float
        Learning rate for the model.
    pretrained_model_path: str
        Path to the pretrained CustomMoLFormer model.
    nb_epoch: int
        Number of epochs for training.
    logger: logging.Logger
        Logger for the experiment.

    Returns
    -------
    avg_score: float
        Average score of the triplicate runs.
    """
    scores = []
    train_dataset = dc.data.DiskDataset(
        f'../../data/featurized_datasets/{splits_name}/dummy_featurized/{dataset}/train'
    )
    valid_dataset = dc.data.DiskDataset(
        f'../../data/featurized_datasets/{splits_name}/dummy_featurized/{dataset}/valid'
    )
    test_dataset = dc.data.DiskDataset(
        f'../../data/featurized_datasets/{splits_name}/dummy_featurized/{dataset}/test'
    )

    for run_id in range(3):
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logger.info(
            f"Starting triplicate run {run_id + 1} for dataset {dataset} at {current_datetime}"
        )
        MoLFormer_model_dir = 'MoLFormer_model_dir'
        os.makedirs(MoLFormer_model_dir, exist_ok=True)
        model_dir = f'./MoLFormer_model_dir/MoLFormer_model_dir_{dataset}_{run_id}_{current_datetime}'
        test_score = run_deepchem_experiment(
            run_id=run_id,
            model_fn=model_fn,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            metric=metric,
            dataset=dataset,
            tasks=tasks,
            model_dir=model_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pretrained_model_path=pretrained_model_path,
            epochs=nb_epoch,
            logger=logger)
        scores.append(test_score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(
        f"Final Triplicate Test Results — Avg {metric.name}: {avg_score:.4f}, Std Dev: {std_score:.4f}"
    )
    return avg_score, std_score


def main():
    """Main function to run the CustomMoLFormer benchmark."""

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--datasets',
        type=str,
        help='comma-separated list of datasets to benchmark',
        default='bbbp,bace,clintox,hiv,tox21,sider')
    argparser.add_argument('--splits_name',
                           type=str,
                           help='name of the splits to use for the datasets',
                           default='molformer_splits')
    argparser.add_argument('--batch_size',
                           type=int,
                           help='batch size for training',
                           default=32)
    argparser.add_argument('--learning_rate',
                           type=float,
                           help='learning rate for training',
                           default=3e-5)
    argparser.add_argument('--epochs',
                           type=int,
                           help='number of epochs for training',
                           default=10)
    argparser.add_argument('--pretrained_model_path',
                           type=str,
                           help='path to the pretrained MoLFormer model',
                           default=None)

    args = argparser.parse_args()

    datasets = args.datasets.split(',')

    if datasets is None:
        raise ValueError("Please provide a list of datasets to benchmark.")
    if not isinstance(datasets, list):
        raise ValueError("Datasets should be provided as a list.")
    if len(datasets) == 0:
        raise ValueError(
            "The list of datasets is empty. Please provide at least one dataset."
        )

    pretrained_model_path = args.pretrained_model_path
    if pretrained_model_path is None:
        raise ValueError("Please provide a path to the pretrained model.")
    if not os.path.exists(pretrained_model_path):
        raise ValueError(
            f"Pretrained model path {pretrained_model_path} does not exist.")

    task_dict = {
        'bbbp': ['p_np'],
        'bace': ['Class'],
        'clintox': ['FDA_APPROVED', 'CT_TOX'],
        'hiv': ['HIV_active'],
        'tox21': [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
            'SR-MMP', 'SR-p53'
        ],
        'sider': [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
            'Product issues', 'Eye disorders', 'Investigations',
            'Musculoskeletal and connective tissue disorders',
            'Gastrointestinal disorders', 'Social circumstances',
            'Immune system disorders',
            'Reproductive system and breast disorders',
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
            'General disorders and administration site conditions',
            'Endocrine disorders', 'Surgical and medical procedures',
            'Vascular disorders', 'Blood and lymphatic system disorders',
            'Skin and subcutaneous tissue disorders',
            'Congenital, familial and genetic disorders',
            'Infections and infestations',
            'Respiratory, thoracic and mediastinal disorders',
            'Psychiatric disorders', 'Renal and urinary disorders',
            'Pregnancy, puerperium and perinatal conditions',
            'Ear and labyrinth disorders', 'Cardiac disorders',
            'Nervous system disorders',
            'Injury, poisoning and procedural complications'
        ]
    }

    # Metric for classification tasks
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    classification_datasets = datasets
    for dataset in classification_datasets:
        if dataset not in task_dict:
            raise ValueError(f"Dataset {dataset} not found in task_dict.")
        logger = setup_logging(dataset=dataset,
                               epochs=args.epochs,
                               batch_size=args.batch_size)
        logger.info(f"Running benchmark for dataset: {dataset}")

        tasks = task_dict[dataset]
        logger.info(
            f"dataset: {dataset}, tasks: {tasks}, epochs: {args.epochs}")
        print("learning rate:", args.learning_rate)
        triplicate_benchmark_dc(dataset=dataset,
                                splits_name=args.splits_name,
                                model_fn=model_fn,
                                metric=metric,
                                tasks=tasks,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate,
                                pretrained_model_path=pretrained_model_path,
                                nb_epoch=args.epochs,
                                logger=logger)


if __name__ == "__main__":
    main()
