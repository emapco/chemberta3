import os
import torch
import argparse
import logging
import numpy as np
import deepchem as dc
import random
import shutil
from datetime import datetime
from deepchem.models.torch_models import Chemberta
from typing import List, Tuple, Callable


# Set seeds
def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed to set.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(dataset: str, epochs: int,
                  batch_size: int) -> logging.Logger:
    """Set up logging for the experiment.

    Parameters
    ----------
    dataset : str
        Name of the dataset being used.
    epochs : int
        Number of epochs for training.
    batch_size : int
        Batch size used for training.

    Returns
    -------
    logger : logging.Logger
        Configured logger for the experiment.
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


def model_fn(tasks: List, model_dir: str, learning_rate: float,
             batch_size: int, pretrained_model_path: str) -> Chemberta:
    # training model
    """Create a ChemBERTa model for classification tasks.

    Parameters
    ----------
    tasks : List
        List of tasks for the model.
    model_dir : str
        Directory to save the model.
    learning_rate : float
        Learning rate for the model.
    batch_size : int
        Batch size for the model.
    pretrained_model_path : str
        Path to the pretrained ChemBERTa model.

    Returns
    -------
    model : Chemberta
        ChemBERTa model for classification tasks.
    """

    finetune_model = Chemberta(task='classification',
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
                            logger: logging.Logger = None) -> float:
    """Run a single experiment with the given parameters.

    Parameters
    ----------
    run_id : int
        ID of the current run.
    model_fn : function
        Function to create the model.
    train_dataset : dc.data.DiskDataset
        Training dataset.
    valid_dataset : dc.data.DiskDataset
        Validation dataset.
    test_dataset : dc.data.DiskDataset
        Test dataset.
    metric : dc.metrics.Metric
        Metric to evaluate the model.
    dataset : str
        Name of the dataset being used.
    tasks : List
        List of tasks for the model.
    model_dir : str
        Directory to save the model.
    batch_size : int
        Batch size for the model.
    learning_rate : float
        Learning rate for the model.
    pretrained_model_path : str
        Path to the pretrained ChemBERTa model.
    epochs : int
        Number of epochs for training.
    logger : logging.Logger
        Logger for the experiment.

    Returns
    -------
    test_score : float
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
    dataset : str
        Name of the dataset being used.
    splits_name : str
        Name of the splits to use for the datasets.
    model_fn : function
        Function to create the model.
    metric : dc.metrics.Metric
        Metric to evaluate the model.
    tasks : List
        List of tasks for the model.
    batch_size : int
        Batch size for the model.
    learning_rate : float
        Learning rate for the model.
    pretrained_model_path : str
        Path to the pretrained ChemBERTa model.
    nb_epoch : int
        Number of epochs for training.
    logger : logging.Logger
        Logger for the experiment.

    Returns
    -------
    avg_score : float
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
        chemberta_model_dir = 'chemberta_model_dir'
        os.makedirs(chemberta_model_dir, exist_ok=True)
        model_dir = f'./chemberta_model_dir/chemberta_model_dir_{dataset}_{run_id}_{current_datetime}'
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
    """Main function to run the ChemBERTa benchmark."""

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
                           help='path to the pretrained ChemBERTa model',
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
