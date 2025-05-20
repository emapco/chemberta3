import os
import torch
import logging
import numpy as np
import argparse
import random
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from functools import partial
from typing import Callable, Tuple, List


# Set seeds
def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed: int
        the seed value to set for random number generation.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(dataset: str, splits_name: str) -> logging.Logger:
    """
    Set up logging for the experiment.

    Parameters
    ----------
    dataset: str
        the name of the dataset being used.
    splits_name: str
        the name of the splits being used.

    Returns
    -------
    logger: logging.Logger
        the configured logger.
    """
    # Create a directory for logs if it doesn't exist
    log_dir = f'logs_{splits_name}_rf'
    os.makedirs(log_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        log_dir, f"rf_{splits_name}_run_{dataset}_{datetime_str}.log")

    logger = logging.getLogger(f"logs_{splits_name}_rf_{dataset}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def model_fn(bootstrap: bool, criterion: str, min_samples_split: int,
             n_estimators: int, model_dir: str,
             tasks: List) -> dc.models.SklearnModel:
    """
    Function to create and return a RandomForestClassifier model wrapped in a DeepChem SklearnModel.

    Parameters
    ----------
    bootstrap: bool
        whether bootstrap samples are used when building trees.
    criterion: str
        the function to measure the quality of a split.
    min_samples_split: int
        the minimum number of samples required to split an internal node.
    n_estimators: int
        the number of trees in the forest.
    model_dir: str
        If specified the model will be stored in this directory. Else, a
        temporary directory will be used.
    tasks: List
        list of tasks

    Returns
    -------
    model: dc.models.SklearnModel
        a DeepChem model wrapping the RandomForestClassifier.
    """
    # training model
    params = {
        "bootstrap": bootstrap,
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "n_estimators": n_estimators
    }

    def rf_model_builder(model_dir, params):
        sklearn_model = RandomForestClassifier(n_jobs=-1, **params)
        return dc.models.SklearnModel(sklearn_model, model_dir)

    model = dc.models.SingletaskToMultitask(
        tasks, partial(rf_model_builder, params=params))

    return model


def run_deepchem_experiment(run_id: int,
                            model_fn: Callable,
                            train_dataset: dc.data.DiskDataset,
                            valid_dataset: dc.data.DiskDataset,
                            test_dataset: dc.data.DiskDataset,
                            metric: dc.metrics.Metric,
                            bootstrap: bool,
                            criterion: str,
                            min_samples_split: int,
                            n_estimators: int,
                            model_dir: str,
                            tasks: List,
                            logger: logging.Logger = None) -> float:
    """
    Run a DeepChem experiment with the specified model and datasets.

    Parameters
    ----------
    run_id: int
        the ID of the current run.
    model_fn: Callable
        function to create the model.
    train_dataset: dc.data.DiskDataset
        the training dataset.
    valid_dataset: dc.data.DiskDataset
        the validation dataset.
    test_dataset: dc.data.DiskDataset
        the test dataset.
    metric: dc.metrics.Metric
        the metric to evaluate the model.
    bootstrap: bool
        whether bootstrap samples are used when building trees.
    criterion: str
        the function to measure the quality of a split.
    min_samples_split: int
        the minimum number of samples required to split an internal node.
    n_estimators: int
        the number of trees in the forest.
    model_dir: str
        If specified the model will be stored in this directory. Else, a
        temporary directory will be used.
    tasks: List
        List of tasks
    logger: logging.Logger
        logger for logging messages.

    Returns
    -------
    test_score: float
        the test score of the model.
    """

    set_seed(run_id)

    model = model_fn(bootstrap=bootstrap,
                     criterion=criterion,
                     min_samples_split=min_samples_split,
                     n_estimators=n_estimators,
                     model_dir=model_dir,
                     tasks=tasks)

    # Get current datetime and format it
    loss = model.fit(train_dataset)
    scores = model.evaluate(valid_dataset, [metric])
    val_score = scores[metric.name]
    test_scores = model.evaluate(test_dataset, [metric])
    test_score = test_scores[metric.name]
    logger.info(f"Valid {metric.name}: {val_score:.4f}")
    logger.info(f"Test {metric.name}: {test_score:.4f}")

    return test_score


def triplicate_benchmark_dc(
        dataset: str,
        splits_name: str,
        model_fn: Callable,
        metric: dc.metrics.Metric,
        bootstrap: bool,
        criterion: str,
        min_samples_split: int,
        n_estimators: int,
        tasks: List,
        logger: logging.Logger = None) -> Tuple[float, float]:
    """
    Run a triplicate benchmark for the specified dataset.

    Parameters
    ----------
    dataset: str
        the name of the dataset being used.
    splits_name: str
        the name of the splits being used.
    model_fn: Callable
        function to create the model.
    metric: dc.metrics.Metric
        the metric to evaluate the model.
    bootstrap: bool
        whether bootstrap samples are used when building trees.
    criterion: str
        the function to measure the quality of a split.
    min_samples_split: int
        the minimum number of samples required to split an internal node.
    n_estimators: int
        the number of trees in the forest.
    tasks: List
        List of classification tasks
    logger: logging.Logger
        logger for logging messages.

    Returns
    -------
    avg_score: float
        the average score across the triplicate runs.
    std_score: float
        the standard deviation of the scores across the triplicate runs.
    """
    scores = []
    train_dataset = dc.data.DiskDataset(
        f'../../data/featurized_datasets/{splits_name}/ecfp_featurized_size1024/{dataset}/train'
    )
    valid_dataset = dc.data.DiskDataset(
        f'../../data/featurized_datasets/{splits_name}/ecfp_featurized_size1024/{dataset}/valid'
    )
    test_dataset = dc.data.DiskDataset(
        f'../../data/featurized_datasets/{splits_name}/ecfp_featurized_size1024/{dataset}/test'
    )

    for run_id in range(3):
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logger.info(
            f"Starting triplicate run {run_id + 1} for {splits_name} dataset {dataset} at {current_datetime}"
        )
        model_dir = f'./rf_{splits_name}_model_dir/rf_model_dir_{dataset}_{run_id}_{current_datetime}'
        test_score = run_deepchem_experiment(
            run_id=run_id,
            splits_name=splits_name,
            model_fn=model_fn,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            metric=metric,
            dataset=dataset,
            bootstrap=bootstrap,
            criterion=criterion,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
            model_dir=model_dir,
            tasks=tasks,
            logger=logger)
        scores.append(test_score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(
        f"Final Triplicate Test Results — Avg {metric.name}: {avg_score:.4f}, Std Dev: {std_score:.4f}"
    )
    return avg_score, std_score


def main():
    """Main function to run the rf classification benchmark."""

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--datasets',
        type=str,
        help='comma-separated list of datasets to benchmark',
        default='sider,tox21,clintox')
    argparser.add_argument('--splits_name',
                           type=str,
                           help='name of the splits to use for the datasets',
                           default='molformer_splits')
    argparser.add_argument(
        '--bootstrap',
        type=bool,
        help='whether bootstrap samples are used when building trees',
        default=False)
    argparser.add_argument(
        '--criterion',
        type=str,
        help='the function to measure the quality of a split',
        default='gini')
    argparser.add_argument(
        '--min_samples_split',
        type=int,
        help='the minimum number of samples required to split an internal node',
        default=2)
    argparser.add_argument('--n_estimators',
                           type=int,
                           help='the number of trees in the forest',
                           default=100)

    args = argparser.parse_args()
    datasets = args.datasets.split(',')

    if datasets is None:
        raise ValueError("Please provide a list of datasets to benchmark.")
    if not isinstance(datasets, list):
        raise ValueError("Datasets should be provided as a list.")
    if len(datasets) == 0:
        raise ValueError("The list of datasets is empty. \
                Please provide at least one dataset.")

    task_dict = {
        'clintox': ['FDA_APPROVED', 'CT_TOX'],
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

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    classification_datasets = datasets

    for dataset in classification_datasets:
        if dataset not in task_dict:
            raise ValueError(f"Dataset {dataset} not found in task_dict.")
        logger = setup_logging(dataset=dataset, splits_name=args.splits_name)
        logger.info(
            f"Running benchmark for dataset: {dataset}, {args.splits_name}, bootstrap: {args.bootstrap}, criterion: {args.criterion},\
min_samples_split: {args.min_samples_split}, n_estimators: {args.n_estimators}"
        )

        tasks = task_dict[dataset]
        triplicate_benchmark_dc(dataset=dataset,
                                splits_name=args.splits_name,
                                model_fn=model_fn,
                                metric=metric,
                                bootstrap=args.bootstrap,
                                criterion=args.criterion,
                                min_samples_split=args.min_samples_split,
                                n_estimators=args.n_estimators,
                                tasks=tasks,
                                logger=logger)


if __name__ == "__main__":
    main()
