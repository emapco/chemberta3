# type: ignore
import argparse
import gc
import logging
import os
import random
import shutil
from collections.abc import Callable
from datetime import datetime

import deepchem as dc
import numpy as np
import torch
from deepchem.models.torch_models import Chemberta
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator

from modchembert.patch import patch_deep_chem_hf_model

patch_deep_chem_hf_model()

transformers_mapping = {
    "balancing": TransformerGenerator(dc.trans.BalancingTransformer),
    "normalization": TransformerGenerator(dc.trans.NormalizationTransformer, transform_y=True),
    "minmax": TransformerGenerator(dc.trans.MinMaxTransformer, transform_y=True),
    "clipping": TransformerGenerator(dc.trans.ClippingTransformer, transform_y=True),
    "log": TransformerGenerator(dc.trans.LogTransformer, transform_y=True),
}


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


def setup_logging(dataset: str, epochs: int, batch_size: int, splits_name: str) -> logging.Logger:
    """Set up logging for the experiment.

    Parameters
    ----------
    dataset: str
        Name of the dataset being used.
    splits_name: str
        Name of the splits to use for the datasets.
    epochs: int
        Number of epochs for training.
    batch_size: int
        Batch size used for training.

    Returns
    -------
    logger: logging.Logger
        Configured logger for the experiment.
    """

    # Create a directory for logs if it doesn't exist
    log_dir = "logs_chemberta_regression"
    os.makedirs(log_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        log_dir, f"chemberta_{splits_name}_run_{dataset}_epochs{epochs}_batch_size{batch_size}_{datetime_str}.log"
    )

    logger = logging.getLogger(f"logs_chemberta_{dataset}_epochs{epochs}_batch_size{batch_size}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def transform_splits(
    train_dataset: dc.data.DiskDataset,
    valid_dataset: dc.data.DiskDataset,
    test_dataset: dc.data.DiskDataset,
    transformer_generators: list,
) -> tuple[tuple, list]:
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
    transformer_generators: list[Union[str, TransformerGenerator]]
        A list of transformer generator objects or string names representing them. If a string is
        passed, it is resolved using the `transformers_mapping` dictionary. Each generator should
        implement a `create_transformer` method.

    Returns
    -------
    tuple[tuple[Dataset, Dataset, Dataset], list[Transformer]]
        A tuple where:
        - The first element is another tuple containing the transformed (train, valid, test) datasets.
        - The second element is the list of fitted transformer objects.

    Notes
    -----
    - All transformations are fitted only on the training dataset.
    - Assumes that each transformer object has `create_transformer()` and `transform()` methods.
    """

    transformers = [transformers_mapping[t.lower()] if isinstance(t, str) else t for t in transformer_generators]

    transformer_dataset = train_dataset
    transformers = [t.create_transformer(transformer_dataset) for t in transformers]
    for transformer in transformers:
        train_dataset = transformer.transform(train_dataset)
        valid_dataset = transformer.transform(valid_dataset)
        test_dataset = transformer.transform(test_dataset)
    return (train_dataset, valid_dataset, test_dataset), transformers


def model_fn(
    tasks: list, model_dir: str, learning_rate: float, batch_size: int, pretrained_model_path: str
) -> Chemberta:
    # training model
    """Create a ChemBERTa model for regression tasks.

    Parameters
    ----------
    tasks: list
        list of tasks for the model.
    model_dir: str
        Directory to save the model.
    learning_rate: float
        Learning rate for the model.
    batch_size: int
        Batch size for the model.
    pretrained_model_path: str
        Path to the pretrained ChemBERTa model.

    Returns
    -------
    model: Chemberta
        ChemBERTa model for regression tasks.
    """
    kwargs = {"num_labels": len(tasks)}
    finetune_model = Chemberta(
        task="regression",
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_tasks=len(tasks),
        model_dir=model_dir,
        tokenizer_path=pretrained_model_path,
    )
    finetune_model.load_from_pretrained(pretrained_model_path, from_hf_checkpoint=True, kwargs=kwargs)
    return finetune_model


def run_deepchem_experiment(
    run_id: int,
    splits_name: str,
    model_fn: Callable,
    train_dataset: dc.data.DiskDataset,
    valid_dataset: dc.data.DiskDataset,
    test_dataset: dc.data.DiskDataset,
    metric: dc.metrics.Metric,
    dataset: str,
    tasks: list,
    model_dir: str,
    batch_size: int,
    learning_rate: float,
    pretrained_model_path: str,
    epochs: int = 100,
    transformer_generators: list = None,
    logger: logging.Logger = None,
) -> float:
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
    tasks: list
        list of tasks for the model.
    model_dir: str
        Directory to save the model.
    batch_size: int
        Batch size for the model.
    learning_rate: float
        Learning rate for the model.
    pretrained_model_path: str
        Path to the pretrained ChemBERTa model.
    epochs: int
        Number of epochs for training.
    logger: logging.Logger
        Logger for the experiment.

    Returns
    -------
    test_score: float
        Test score of the model.
    """
    if transformer_generators is None:
        transformer_generators = []

    set_seed(run_id)
    model = model_fn(
        tasks=tasks,
        model_dir=model_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pretrained_model_path=pretrained_model_path,
    )
    best_score = np.inf

    # Get current datetime and format it
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_model_dir = f"checkpoints_{dataset}/best_model_epochs{epochs}_run{run_id}_{current_datetime}"
    os.makedirs(f"checkpoints_{dataset}", exist_ok=True)

    if transformer_generators:
        (train_dataset, valid_dataset, test_dataset), transformers = transform_splits(
            train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            transformer_generators=transformer_generators,
        )
    else:
        transformers = []

    for epoch in range(epochs):
        loss = model.fit(train_dataset, nb_epoch=1, restore=epoch > 0, max_checkpoints_to_keep=1)
        scores = model.evaluate(dataset=valid_dataset, metrics=[metric], transformers=transformers)
        val_score = scores[metric.name]

        logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss:.4f} | Val {metric.name}: {val_score:.4f}")

        if epoch % 10 == 0 and epoch > 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if val_score < best_score:
            best_score = val_score

            # Save current checkpoint as best model
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
            logger.info(f"Global step of best model: {model._global_step}")
            shutil.copytree(model.model_dir, best_model_dir)

            logger.info(f"Best model saved at epoch {epoch + 1} with val {metric.name}: {val_score:.4f}")

    # Load best checkpoint before evaluating on test set
    model.restore(model_dir=best_model_dir)
    test_scores = model.evaluate(dataset=test_dataset, metrics=[metric], transformers=transformers)
    test_score = test_scores[metric.name]
    logger.info(f"Test {metric.name}: {test_score:.4f}")

    return test_score


def triplicate_benchmark_dc(
    dataset: str,
    splits_name: str,
    model_fn: Callable,
    metric: dc.metrics.Metric,
    tasks: list,
    batch_size: int,
    learning_rate: float,
    pretrained_model_path: str,
    transformer_generators: list,
    nb_epoch: int = 100,
    logger: logging.Logger = None,
) -> tuple[float, float]:
    """Run a triplicate benchmark for the given dataset.

    Parameters
    ----------
    dataset: str
        Name of the dataset being used.
    splits_name: str
        Name of the splits to use for the datasets.
    model_fn: function
        Function to create the model.
    metric: dc.metrics.Metric
        Metric to evaluate the model.
    tasks: list
        list of tasks for the model.
    batch_size: int
        Batch size for the model.
    learning_rate: float
        Learning rate for the model.
    pretrained_model_path: str
        Path to the pretrained ChemBERTa model.
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
    train_dataset_address = f"../../data/featurized_datasets/{splits_name}/dummy_featurized/{dataset}/train"
    train_dataset = dc.data.DiskDataset(train_dataset_address)
    valid_dataset = dc.data.DiskDataset(
        f"../../data/featurized_datasets/{splits_name}/dummy_featurized/{dataset}/valid"
    )
    test_dataset = dc.data.DiskDataset(f"../../data/featurized_datasets/{splits_name}/dummy_featurized/{dataset}/test")
    logger.info(f"train_dataset: {train_dataset_address}")

    for run_id in range(3):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"Starting triplicate run {run_id + 1} for dataset {dataset} at {current_datetime}")
        chemberta_model_dir = "chemberta_model_dir"
        os.makedirs(chemberta_model_dir, exist_ok=True)
        model_dir = f"./chemberta_model_dir/chemberta_model_dir_{dataset}_{run_id}_{current_datetime}"
        test_score = run_deepchem_experiment(
            run_id=run_id,
            splits_name=splits_name,
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
            transformer_generators=transformer_generators,
            logger=logger,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        scores.append(test_score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Final Triplicate Test Results — Avg {metric.name}: {avg_score:.4f}, Std Dev: {std_score:.4f}")
    return avg_score, std_score


def main():
    """Main function to run the ChemBERTa benchmark."""

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--datasets", type=str, help="comma-separated list of datasets to benchmark", default="esol,freesolv,lipo"
    )
    argparser.add_argument(
        "--splits_name", type=str, help="name of the splits to use for the datasets", default="molformer_splits"
    )
    argparser.add_argument(
        "--transform", action="store_true", help="Select True, to apply transformation to dataset", default=False
    )
    argparser.add_argument("--batch_size", type=int, help="batch size for training", default=32)
    argparser.add_argument("--learning_rate", type=float, help="learning rate for training", default=3e-5)
    argparser.add_argument("--epochs", type=int, help="number of epochs for training", default=10)
    argparser.add_argument(
        "--pretrained_model_path", type=str, help="path to the pretrained ChemBERTa model", default=None
    )

    args = argparser.parse_args()

    datasets = args.datasets.split(",")

    if datasets is None:
        raise ValueError("Please provide a list of datasets to benchmark.")
    if not isinstance(datasets, list):
        raise ValueError("Datasets should be provided as a list.")
    if len(datasets) == 0:
        raise ValueError("The list of datasets is empty. Please provide at least one dataset.")

    pretrained_model_path = args.pretrained_model_path
    if pretrained_model_path is None:
        raise ValueError("Please provide a path to the pretrained model.")

    task_dict = {
        "delaney": ["measured_log_solubility_in_mols_per_litre"],
        "freesolv": ["y"],
        "lipo": ["exp"],
        "clearance": ["target"],
        "bace_regression": ["pIC50"],
    }

    transformer_generators = {
        "delaney": ["normalization"],
        "freesolv": ["normalization"],
        "lipo": ["normalization"],
        "clearance": ["log"],
        "bace_regression": ["normalization"],
    }

    # Metric for regression tasks
    metric = dc.metrics.Metric(dc.metrics.rms_score)
    regression_datasets = datasets

    for dataset in regression_datasets:
        transformers = transformer_generators[dataset] if args.transform is True else []
        if dataset not in task_dict:
            raise ValueError(f"Dataset {dataset} not found in task_dict.")
        logger = setup_logging(
            dataset=dataset, epochs=args.epochs, batch_size=args.batch_size, splits_name=args.splits_name
        )
        logger.info(f"Running benchmark for dataset: {dataset}")

        tasks = task_dict[dataset]
        logger.info(
            f"dataset: {dataset}, tasks: {tasks}, epochs: {args.epochs}, "
            f"splits_name: {args.splits_name}, learning rate: {args.learning_rate}, transform: {args.transform}"
        )
        triplicate_benchmark_dc(
            dataset=dataset,
            splits_name=args.splits_name,
            model_fn=model_fn,
            metric=metric,
            tasks=tasks,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pretrained_model_path=pretrained_model_path,
            nb_epoch=args.epochs,
            transformer_generators=transformers,
            logger=logger,
        )


if __name__ == "__main__":
    main()
