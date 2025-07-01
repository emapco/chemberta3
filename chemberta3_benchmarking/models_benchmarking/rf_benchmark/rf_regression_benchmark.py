import os
import torch
import joblib
import logging
import numpy as np
import argparse
import random
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from typing import Callable, Tuple, List
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator


transformers_mapping = {
    'balancing':
        TransformerGenerator(dc.trans.BalancingTransformer),
    'normalization':
        TransformerGenerator(dc.trans.NormalizationTransformer,
                            transform_y=True),
    'minmax':
        TransformerGenerator(dc.trans.MinMaxTransformer, transform_y=True),
    'clipping':
        TransformerGenerator(dc.trans.ClippingTransformer, transform_y=True),
    'log':
        TransformerGenerator(dc.trans.LogTransformer, transform_y=True)
}


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
    log_path = os.path.join(log_dir, f"rf_{splits_name}_run_{dataset}_{datetime_str}.log")

    logger = logging.getLogger(f"logs_{splits_name}_rf_{dataset}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    return logger


def transform_splits(train_dataset: dc.data.DiskDataset,
                     valid_dataset: dc.data.DiskDataset,
                     test_dataset: dc.data.DiskDataset,
                     transformer_generators: List) -> Tuple[Tuple, List]:
    """
    Applies a sequence of data transformations to train, validation, and test datasets.

    This function first initializes transformers using the provided transformer generator 
    objects or string names (which are mapped via `transformers_mapping`). The transformers 
    are fitted only on the training dataset and then applied to all three splits to ensure 
    consistent transformation.

    Parameters
    ----------
    train_dataset: dc.data.DiskDataset
        The training dataset to fit and transform.
    valid_dataset: dc.data.DiskDataset
        The validation dataset to be transformed using the same transformers as the training set.
    test_dataset: dc.data.DiskDataset
        The test dataset to be transformed using the same transformers as the training set.
    transformer_generators: List[Union[str, TransformerGenerator]]
        A list of transformer generator objects or string names representing them. If a string is 
        passed, it is resolved using the `transformers_mapping` dictionary. Each generator should 
        implement a `create_transformer` method.

    Returns
    -------
    Tuple[Tuple[Dataset, Dataset, Dataset], List[Transformer]]
        A tuple where:
        - The first element is another tuple containing the transformed (train, valid, test) datasets.
        - The second element is the list of fitted transformer objects.

    Notes
    -----
    - All transformations are fitted only on the training dataset.
    - Assumes that each transformer object has `create_transformer()` and `transform()` methods.
    """

    transformers = [
                transformers_mapping[t.lower()] if isinstance(t, str) else t
                for t in transformer_generators
            ]

    transformer_dataset = train_dataset
    transformers = [
        t.create_transformer(transformer_dataset) for t in transformers
    ]
    for transformer in transformers:
        train_dataset = transformer.transform(train_dataset)
        valid_dataset = transformer.transform(valid_dataset)
        test_dataset = transformer.transform(test_dataset)
    return (train_dataset, valid_dataset, test_dataset), transformers


def model_fn(bootstrap: bool, 
             criterion: str, 
             min_samples_split: int, 
             n_estimators: int,
             model_dir: str) -> dc.models.SklearnModel:
    """
    Function to create and return a RandomForestRegressor model wrapped in a DeepChem SklearnModel.

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

    Returns
    -------
    model: dc.models.SklearnModel
        a DeepChem model wrapping the RandomForestRegressor.
    """
    # training model
    params = {
        "bootstrap": bootstrap,
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "n_estimators": n_estimators
    }
    sklearn_model = RandomForestRegressor(**params, n_jobs=-1)
    model = dc.models.SklearnModel(sklearn_model, model_dir=model_dir)

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
                            transformer_generators: List,
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
                     model_dir=model_dir)

    if transformer_generators:
        (train_dataset, valid_dataset, test_dataset), transformers = transform_splits(train_dataset,
                                                                                  valid_dataset=valid_dataset,
                                                                                  test_dataset=test_dataset,
                                                                                  transformer_generators=transformer_generators)
    else:
        transformers = []
    loss = model.fit(train_dataset)
    model.save()
    scores = model.evaluate(dataset=valid_dataset, metrics=[metric], transformers=transformers)
    val_score = scores[metric.name]
    test_scores = model.evaluate(dataset=test_dataset, metrics=[metric], transformers=transformers)
    test_score = test_scores[metric.name]
    logger.info(f"Valid {metric.name}: {val_score:.4f}")
    logger.info(f"Test {metric.name}: {test_score:.4f}")

    return test_score


def triplicate_benchmark_dc(dataset: str,
                            splits_name: str,
                            model_fn: Callable,
                            metric: dc.metrics.Metric,
                            bootstrap: bool,
                            criterion: str,
                            min_samples_split: int,
                            n_estimators: int,
                            transformer_generators: List,
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

    train_dataset_address = f'../../data/featurized_datasets/{splits_name}/ecfp_featurized_size1024/{dataset}/train'
    train_dataset = dc.data.DiskDataset(train_dataset_address)
    valid_dataset = dc.data.DiskDataset(f'../../data/featurized_datasets/{splits_name}/ecfp_featurized_size1024/{dataset}/valid')
    test_dataset = dc.data.DiskDataset(f'../../data/featurized_datasets/{splits_name}/ecfp_featurized_size1024/{dataset}/test')

    logger.info(f"train_dataset: {train_dataset_address}")

    for run_id in range(3):
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logger.info(f"Starting triplicate run {run_id + 1} for {splits_name} dataset {dataset} at {current_datetime}")
        rf_model_dir = f'rf_{splits_name}_model_dir'
        os.makedirs(rf_model_dir, exist_ok=True)
        model_dir = f'./rf_{splits_name}_model_dir/rf_model_dir_{dataset}_{run_id}_{current_datetime}'
        test_score = run_deepchem_experiment(
            run_id=run_id, splits_name=splits_name,
            model_fn=model_fn, train_dataset=train_dataset, 
            valid_dataset=valid_dataset, test_dataset=test_dataset,
            metric=metric, dataset=dataset, bootstrap=bootstrap, criterion=criterion,
            min_samples_split=min_samples_split, n_estimators=n_estimators,
            model_dir=model_dir, transformer_generators=transformer_generators, logger=logger
        )
        scores.append(test_score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Final Triplicate Test Results — Avg {metric.name}: {avg_score:.4f}, Std Dev: {std_score:.4f}")
    return avg_score, std_score


def main():
    """Main function to run the rf regression benchmark."""

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
    argparser.add_argument('--transform',
                        action='store_true',
                        help='Select True, to apply transformation to dataset',
                        default=False)
    argparser.add_argument('--bootstrap',
                           type=bool,
                           help='whether bootstrap samples are used when building trees',
                           default=False)
    argparser.add_argument('--criterion',
                           type=str,
                           help='the function to measure the quality of a split',
                           default='squared_error')
    argparser.add_argument('--min_samples_split',
                           type=int,
                           help='the minimum number of samples required to split an internal node',
                           default=2)
    argparser.add_argument('--n_estimators',
                           type=int,
                           help='the number of trees in the forest',
                           default=10)

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
        'esol': ['measured_log_solubility_in_mols_per_litre'],
        'freesolv': ['y'],
        'lipo': ['exp'],
        'clearance': ['target'],
        'bace_regression': ['pIC50']
    }

    transformer_generators = {
        'esol': ['normalization'],
        'freesolv': ['normalization'],
        'lipo': ['normalization'],
        'clearance': ['log'],
        'bace_regression': ['normalization'],
    }

    metric = dc.metrics.Metric(dc.metrics.rms_score)
    regression_datasets = datasets

    for dataset in regression_datasets:

        if args.transform is True:
            transformers = transformer_generators[dataset]
        else:
            transformers = []
        if dataset not in task_dict:
            raise ValueError(f"Dataset {dataset} not found in task_dict.")
        logger = setup_logging(dataset=dataset,
                               splits_name=args.splits_name)
        tasks = task_dict[dataset]
        logger.info(f"Running benchmark for dataset: {dataset}, {args.splits_name}, bootstrap: {args.bootstrap}, criterion: {args.criterion},\
min_samples_split: {args.min_samples_split}, n_estimators: {args.n_estimators}, task: {tasks}")

        
        triplicate_benchmark_dc(dataset=dataset,
                                splits_name=args.splits_name,
                                model_fn=model_fn,
                                metric=metric,
                                bootstrap=args.bootstrap,
                                criterion=args.criterion,
                                min_samples_split=args.min_samples_split,
                                n_estimators=args.n_estimators,
                                transformer_generators=transformers,
                                logger=logger)


if __name__ == "__main__":
    main()
