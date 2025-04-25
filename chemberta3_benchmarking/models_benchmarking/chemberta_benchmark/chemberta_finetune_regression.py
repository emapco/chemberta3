import os
import torch
import argparse
import logging
import numpy as np
import deepchem as dc
from deepchem.models.torch_models import Chemberta
import random
from typing import List, Tuple, Callable, Optional
from datetime import datetime
import shutil


# Set seeds
def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
        seed : int
            The seed value to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(dataset: str, epochs: int,
                  batch_size: int) -> logging.Logger:
    """
    Set up logging for the experiment.

    Parameters
    ----------
        dataset : int
            The name of the dataset.
        epochs : int
            The number of epochs.
        batch_size : int
            The batch size.
    Returns
    -------
        logger : logging.Logger
            The configured logger.
    """
    # Create a directory for logs if it doesn't exist
    log_dir = 'logs_chemberta'
    os.makedirs(log_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        log_dir,
        f"chemberta_molformer_splits_run_{dataset}_epochs{epochs}_batch_size{batch_size}_{datetime_str}.log"
    )

    logger = logging.getLogger(
        f"logs_chemberta_{dataset}_epochs{epochs}_batch_size{batch_size}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def model_fn(tasks: List, model_dir: str, batch_size: int,
             learning_rate: float, pretrained_model_path: str) -> Chemberta:
    """
    Create a Chemberta model for regression tasks.

    Parameters
    ----------
        tasks : List
            List of tasks for the model.
        model_dir : str
            Directory to save the model.
        batch_size : int
            Batch size for training.
        learning_rate : float
            Learning rate for training.
        pretrained_model_path : str
            Path to the pretrained ChemBERTa model.
    Returns
    -------
        finetune_model : Chemberta
            The Chemberta model for regression tasks.
    """
    # training model

    finetune_model = Chemberta(task='regression',
                               learning_rate=learning_rate,
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
                            logger: Optional[logging.Logger] = None) -> float:
    """
    Run a DeepChem experiment for regression tasks.

    Parameters
    ----------
        run_id : int
            The ID of the run.
        model_fn : function
            Function to create the model.
        train_dataset : deepchem.data.Dataset
            Training dataset.
        valid_dataset : deepchem.data.Dataset
            Validation dataset.
        test_dataset : deepchem.data.Dataset
            Test dataset.
        metric : deepchem.metrics.Metric
            Metric for evaluation.
        dataset : str
            Name of the dataset.
        tasks : List
            List of tasks for the model.
        model_dir : str
            Directory to save the model.
        batch_size : int
            Batch size for training.
        learning_rate : float
            Learning rate for training.
        pretrained_model_path : str
            Path to the pretrained ChemBERTa model.
        epochs : int, Optional
            Number of epochs for training (default is 100).
        logger : logging.Logger, Optional
            Logger for logging (default is None).
    Returns
    -------
        test_score : float
            The test score of the model.
    """

    set_seed(run_id)
    model = model_fn(tasks, model_dir, batch_size, learning_rate,
                     pretrained_model_path)
    best_score = np.inf

    # Get current datetime and format it
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    best_model_dir = f"checkpoints_{dataset}/best_model_epochs{epochs}_run{run_id}_{current_datetime}"
    os.makedirs(f"checkpoints_{dataset}", exist_ok=True)

    for epoch in range(epochs):

        loss = model.fit(train_dataset,
                         nb_epoch=1,
                         restore=epoch > 0,
                         max_checkpoints_to_keep=1)
        scores = model.evaluate(valid_dataset, [metric])
        val_score = scores[metric.name]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Val {metric.name}: {val_score:.4f}"
        )

        if val_score < best_score:
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


def triplicate_benchmark_dc(dataset: str,
                            splits_name: str,
                            model_fn: Callable,
                            metric: dc.metrics.Metric,
                            tasks: List,
                            batch_size: int,
                            learning_rate: float,
                            pretrained_model_path: str,
                            nb_epoch: int = 100,
                            logger: Optional[logging.Logger] = None) -> Tuple:
    """
    Run a triplicate benchmark for the Chemberta model.

    Parameters
    ----------
        dataset : str
            The name of the dataset.
        splits_name : str
            The name of the splits to use for the datasets.
        model_fn : function
            Function to create the model.
        metric : deepchem.metrics.Metric
            Metric for evaluation.
        tasks : List
            List of tasks for the model.
        batch_size : int
            Batch size for training.
        learning_rate : float
            Learning rate for training.
        pretrained_model_path : str
            Path to the pretrained ChemBERTa model.
        nb_epoch : int, Optional
            Number of epochs for training (default is 100).
        logger : logging.Logger, Optional
            Logger for logging (default is None).
    Returns
    -------
        avg_score : float
            The average score of the triplicate runs.
        std_score : float
            The standard deviation of the scores from the triplicate runs.
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
        chemberta_model_dir = 'chemberta_model_dir'
        os.makedirs(chemberta_model_dir, exist_ok=True)
        model_dir = f'./chemberta_model_dir/chemberta_model_dir_{dataset}_{run_id}_{current_datetime}'
        test_score = run_deepchem_experiment(
            run_id,
            model_fn,
            train_dataset,
            valid_dataset,
            test_dataset,
            metric,
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
    """
    Main function to run the Chemberta regression benchmark.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--datasets',
        type=str,
        help='comma-separated list of datasets to benchmark',
        default='esol,freesolv,lipo')
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
                           help='path to the pretrained ChemBERTa model',
                           default=None)

    args = argparser.parse_args()

    datasets = args.datasets.split(',')

    if datasets is None:
        raise ValueError("Please provide a list of datasets to benchmark.")
    if not isinstance(datasets, List):
        raise ValueError("Datasets should be provided as a list.")
    if len(datasets) == 0:
        raise ValueError(
            "The list of datasets is empty. Please provide at least one dataset."
        )
    epochs = args.epochs
    batch_size = args.batch_size
    pretrained_model_path = args.pretrained_model_path
    if pretrained_model_path is None:
        raise ValueError("Please provide a path to the pretrained model.")
    if not os.path.exists(pretrained_model_path):
        raise ValueError(
            f"Pretrained model path {pretrained_model_path} does not exist.")

    task_dict = {
        'esol': ['measured_log_solubility_in_mols_per_litre'],
        'freesolv': ['expt'],
        'lipo': ['y'],
    }

    metric = dc.metrics.Metric(dc.metrics.rms_score)
    regression_datasets = datasets
    for dataset in regression_datasets:
        if dataset not in task_dict:
            raise ValueError(f"Dataset {dataset} not found in task_dict.")
        logger = setup_logging(dataset=dataset,
                               epochs=epochs,
                               batch_size=batch_size)
        logger.info(f"Running benchmark for dataset: {dataset}")

        tasks = task_dict[dataset]
        logger.info(f"dataset: {dataset}, tasks: {tasks}, epochs: {epochs}")
        triplicate_benchmark_dc(dataset=dataset,
                                splits_name=args.splits_name,
                                model_fn=model_fn,
                                metric=metric,
                                tasks=tasks,
                                batch_size=batch_size,
                                learning_rate=args.learning_rate,
                                pretrained_model_path=pretrained_model_path,
                                nb_epoch=epochs,
                                logger=logger)


if __name__ == "__main__":
    main()
