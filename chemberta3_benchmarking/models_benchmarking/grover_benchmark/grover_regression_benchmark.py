import os
import torch
import logging
import numpy as np
import shutil
import argparse
import deepchem as dc
import random
from datetime import datetime
from typing import Callable, List, Tuple
from deepchem.models.torch_models import GroverModel
from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder)
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
    """
    Setup logging for the benchmarking runs.

    Parameters
    ----------
    dataset: str
        The name of the dataset being used.
    splits_name: str
        Name of the splits to use for the datasets.
    epochs: int
        The number of epochs for training.
    batch_size: int
        The batch size used during training.

    Returns
    -------
    logger: logging.Logger
        Configured logger instance.
    """
    log_dir = f'logs_{splits_name}_grover'
    os.makedirs(log_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"grover_{splits_name}_run_{dataset}_epochs{epochs}_batch_size{batch_size}_{datetime_str}.log")

    logger = logging.getLogger(f"logs_grover_{dataset}_epochs{epochs}_batch_size{batch_size}")
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
    

def model_fn(tasks: List, 
             model_dir: str,
             batch_size: int,
             vocab_data_path: str,
             node_fdim: int,
             edge_fdim: int,
             features_dim: int,
             hidden_size: int,
             functional_group_size: int,
             pretrained_model_path: str,
             learning_rate: float) -> GroverModel: 
    """
    Create and initialize a GroverModel for a regression finetuning task.

    This function builds atom and bond vocabularies using a dataset at `vocab_data_path`,
    restores a pretrained Grover model from the specified checkpoint, and initializes
    a new finetuning model using the pretrained embedding weights.

    Parameters
    ----------
    tasks: List
        List of regression task names (used to determine the number of output heads).
    model_dir: str
        Directory where the model checkpoints and logs will be stored.
    batch_size: int
        Batch size to use during training.
    vocab_data_path: str
        Path to a featurized DeepChem DiskDataset used to build the atom and bond vocabularies.
    node_fdim: int
        Dimension of node (atom) features.
    edge_fdim: int
        Dimension of edge (bond) features.
    features_dim: int
        Dimension of additional molecular features (e.g., circular fingerprints).
    hidden_size: int
        Hidden size used in the Grover message passing network.
    functional_group_size: int
        Number of functional group tokens used in the model.
    pretrained_model_path: str
        Path to the pretrained Grover model checkpoint to load weights from.
    learning_rate: float
        Learning rate for the optimizer during finetuning.

    Returns
    -------
    finetune_model: GroverModel
        A GroverModel initialized for regression and ready for finetuning with the
        pretrained embeddings loaded.
    """
    
    data = dc.data.DiskDataset(vocab_data_path)
    av = GroverAtomVocabularyBuilder()
    av.build(data)
    bv = GroverBondVocabularyBuilder()
    bv.build(data)

    pretrain_model = GroverModel(node_fdim=node_fdim, 
                                 edge_fdim=edge_fdim, 
                                 atom_vocab=av, 
                                 bond_vocab=bv, 
                                 features_dim=features_dim, 
                                 hidden_size=hidden_size, 
                                 functional_group_size=functional_group_size, 
                                 task='pretraining')

    pretrain_model.restore(pretrained_model_path)

    finetune_model = GroverModel(node_fdim=node_fdim, 
                                 edge_fdim=edge_fdim, 
                                 features_dim=features_dim,
                                 hidden_size=hidden_size, 
                                 functional_group_size=functional_group_size, 
                                 task='finetuning', 
                                 mode='regression', 
                                 model_dir=model_dir, 
                                 batch_size=batch_size,
                                 n_tasks=len(tasks),
                                 learning_rate=learning_rate)

    finetune_model.load_from_pretrained(pretrain_model, components=['embedding'])

    return finetune_model


def run_deepchem_experiment(run_id: int,
                            splits_name: str, 
                            model_fn: Callable, 
                            train_dataset: dc.data.DiskDataset, 
                            valid_dataset: dc.data.DiskDataset, 
                            test_dataset: dc.data.DiskDataset, 
                            metric: dc.metrics.Metric, 
                            dataset: str, 
                            tasks: List[str], 
                            model_dir: str,
                            batch_size: int,
                            vocab_data_path: str,
                            node_fdim: int,
                            edge_fdim: int,
                            features_dim: int,
                            hidden_size: int,
                            functional_group_size: int,
                            pretrained_model_path: str,
                            learning_rate: float, 
                            epochs: int = 50,
                            transformer_generators: List = [],
                            logger: logging.Logger = None) -> float:
    """
    Run a DeepChem experiment for regression finetuning.

    Parameters
    ----------
    run_id: int
        Identifier for the current run, used for setting seeds.
    splits_name: str
        Name of the splits to use for the datasets.
    model_fn: Callable
        Function to create the model.
    train_dataset: dc.data.DiskDataset
        Training dataset.
    valid_dataset: dc.data.DiskDataset
        Validation dataset.
    test_dataset: dc.data.DiskDataset
        Test dataset.
    metric: dc.metrics.Metric
        Metric to evaluate the model performance.
    dataset: str
        Name of the dataset being used.
    tasks: List[str]
        List of tasks for the model.
    model_dir: str
        Directory where the model will be saved.
    batch_size: int
        Batch size used during training.
    vocab_data_path : str
        Path to a featurized DeepChem DiskDataset used to build the atom and bond vocabularies.
    node_fdim : int
        Dimension of node (atom) features.
    edge_fdim : int
        Dimension of edge (bond) features.
    features_dim : int
        Dimension of additional molecular features (e.g., circular fingerprints).
    hidden_size : int
        Hidden size used in the Grover message passing network.
    functional_group_size : int
        Number of functional group tokens used in the model.
    pretrained_model_path : str
        Path to the pretrained Grover model checkpoint to load weights from.
    learning_rate : float
        Learning rate for the optimizer during finetuning.
    epochs: int
        Number of epochs for training.
    logger: logging.Logger
        Logger instance for logging the experiment details.

    Returns
    -------
    test_score: float
        The score of the model on the test dataset.
    """
    set_seed(run_id)

    model = model_fn(tasks=tasks, model_dir=model_dir, 
                     batch_size=batch_size, vocab_data_path=vocab_data_path,
                     node_fdim=node_fdim, edge_fdim=edge_fdim,
                     features_dim=features_dim, hidden_size=hidden_size,
                     functional_group_size=functional_group_size,
                     pretrained_model_path=pretrained_model_path,
                     learning_rate=learning_rate)
    best_score = np.inf

    # Get current datetime and format it
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    best_model_dir = f"checkpoints_{splits_name}_{dataset}/best_model_epochs{epochs}_run{run_id}_{current_datetime}"
    os.makedirs(f"checkpoints_{splits_name}_{dataset}", exist_ok=True)

    if transformer_generators:
        (train_dataset, valid_dataset, test_dataset), transformers = transform_splits(train_dataset,
                                                                                  valid_dataset=valid_dataset,
                                                                                  test_dataset=test_dataset,
                                                                                  transformer_generators=transformer_generators)
    else:
        transformers = []

    for epoch in range(epochs):

        loss = model.fit(train_dataset, nb_epoch=1, restore=epoch>0, max_checkpoints_to_keep=1)
        scores = model.evaluate(dataset=valid_dataset, metrics=[metric], transformers=transformers)
        val_score = scores[metric.name]

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Val {metric.name}: {val_score:.4f}")

        if val_score < best_score:
            best_score = val_score
            # model.save_checkpoint(max_checkpoints_to_keep=1)

            # Save current checkpoint as best model
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
            logger.info(f"Global step of best model: {model._global_step}")
            shutil.copytree(model.model_dir, best_model_dir)

            logger.info(f"Best model saved at epoch {epoch+1} with val {metric.name}: {val_score:.4f}")

    # Load best checkpoint before evaluating on test set
    model.restore(model_dir=best_model_dir)
    test_scores = model.evaluate(dataset=test_dataset, metrics=[metric], transformers=transformers)
    test_score = test_scores[metric.name]
    logger.info(f"Test {metric.name}: {test_score:.4f}")

    return test_score


def triplicate_benchmark_dc(dataset: str,
                            splits_name: str,
                            model_fn: Callable, 
                            metric: dc.metrics.Metric, 
                            tasks: List[str], 
                            batch_size: int,
                            vocab_data_path: str,
                            node_fdim: int,
                            edge_fdim: int,
                            features_dim: int,
                            hidden_size: int,
                            functional_group_size: int,
                            learning_rate: float,
                            pretrained_model_path: str,
                            transformer_generators: List,
                            nb_epoch: int = 50, 
                            logger: logging.Logger=None) -> Tuple[float, float]:
    """
    Run a triplicate benchmark for the given dataset using DeepChem.

    Parameters
    ----------
    dataset: str
        Name of the dataset to benchmark.
    splits_name: str
        Name of the splits to use for the datasets.
    model_fn: Callable
        Function to create the model.
    metric: dc.metrics.Metric
        Metric to evaluate the model performance.
    tasks: List[str]
        List of tasks for the model.
    batch_size: int
        Batch size used during training.
    vocab_data_path : str
        Path to a featurized DeepChem DiskDataset used to build the atom and bond vocabularies.
    node_fdim : int
        Dimension of node (atom) features.
    edge_fdim : int
        Dimension of edge (bond) features.
    features_dim : int
        Dimension of additional molecular features (e.g., circular fingerprints).
    hidden_size : int
        Hidden size used in the Grover message passing network.
    functional_group_size : int
        Number of functional group tokens used in the model.
    pretrained_model_path : str
        Path to the pretrained Grover model checkpoint to load weights from.
    learning_rate : float
        Learning rate for the optimizer during finetuning.
    nb_epoch: int
        Number of epochs for training.
    logger: logging.Logger
        Logger instance for logging the experiment details.

    Returns
    -------
    avg_score: float
        Average score of the model across triplicate runs.
    std_score: float
        Standard deviation of the scores across triplicate runs.
    """
    scores = []
    train_dataset_address = f'../../data/featurized_datasets/{splits_name}/grover_featurized/{dataset}/train'
    train_dataset = dc.data.DiskDataset(train_dataset_address)
    valid_dataset = dc.data.DiskDataset(f'../../data/featurized_datasets/{splits_name}/grover_featurized/{dataset}/valid')
    test_dataset = dc.data.DiskDataset(f'../../data/featurized_datasets/{splits_name}/grover_featurized/{dataset}/test')
    logger.info(f"train_dataset: {train_dataset_address}")

    for run_id in range(3):
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logger.info(f"Starting triplicate run {run_id + 1} for dataset {dataset} at {current_datetime}")
        grover_model_dir = f'grover_{splits_name}_model_dir'
        os.makedirs(grover_model_dir, exist_ok=True)
        model_dir = f'./grover_{splits_name}_model_dir/grover_model_dir_{dataset}_{run_id}_{current_datetime}'
        test_score = run_deepchem_experiment(
            run_id=run_id, splits_name=splits_name,
            model_fn=model_fn, train_dataset=train_dataset, 
            valid_dataset=valid_dataset, test_dataset=test_dataset,
            metric=metric, dataset=dataset, tasks=tasks, 
            model_dir=model_dir, batch_size=batch_size, 
            vocab_data_path=vocab_data_path, node_fdim=node_fdim,
            edge_fdim=edge_fdim, features_dim=features_dim,
            hidden_size=hidden_size, 
            functional_group_size=functional_group_size,
            pretrained_model_path=pretrained_model_path,
            learning_rate=learning_rate,
            transformer_generators=transformer_generators,
            epochs=nb_epoch, logger=logger
        )
        scores.append(test_score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Final Triplicate Test Results — Avg {metric.name}: {avg_score:.4f}, Std Dev: {std_score:.4f}")
    return avg_score, std_score


def main():
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
    argparser.add_argument('--batch_size',
                           type=int,
                           help='batch size for training',
                           default=32)
    argparser.add_argument('--vocab_data_path',
                        type=str,
                        help='path to the dataset used to build vocabulary',
                        default=None)
    argparser.add_argument('--node_fdim',
                        type=int,
                        default=151)
    argparser.add_argument('--edge_fdim',
                        type=int,
                        default=165)
    argparser.add_argument('--feature_dim',
                        type=int,
                        default=2048)
    argparser.add_argument('--hidden_size',
                        type=int,
                        default=128)
    argparser.add_argument('--functional_group_size',
                        type=int,
                        default=85)   
    argparser.add_argument('--learning_rate',
                           type=float,
                           help='learning rate for training',
                           default=0.001)
    argparser.add_argument('--epochs',
                           type=int,
                           help='number of epochs for training',
                           default=10)
    argparser.add_argument('--pretrained_model_path',
                           type=str,
                           help='path to the pretrained Infograph model',
                           default=None)

    args = argparser.parse_args()

    datasets = args.datasets.split(',')

    if datasets is None:
        raise ValueError("Please provide a list of datasets to benchmark.")
    if not isinstance(datasets, list):
        raise ValueError("Datasets should be provided as a list.")
    if len(datasets) == 0:
        raise ValueError("The list of datasets is empty. Please provide at least one dataset.")
    epochs = args.epochs
    batch_size = args.batch_size
    pretrained_model_path = args.pretrained_model_path
    if pretrained_model_path is None:
        raise ValueError("Please provide a path to the pretrained model.")
    if not os.path.exists(pretrained_model_path):
        raise ValueError(f"Pretrained model path {pretrained_model_path} does not exist.")

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
        logger = setup_logging(dataset=dataset, epochs=epochs, 
                               batch_size=batch_size, splits_name=args.splits_name)
        logger.info(f"Running benchmark for dataset: {dataset}")

        tasks = task_dict[dataset]
        logger.info(f"dataset: {dataset}, tasks: {tasks}, epochs: {args.epochs}, pretrained_model_path: {args.pretrained_model_path}\
learning_rate: {args.learning_rate}, batch_size: {batch_size}, splits_name: {args.splits_name}, hidden_size: {args.hidden_size}, feature_dim: {args.feature_dim}, transform: {args.transform}")
        triplicate_benchmark_dc(dataset=dataset,
                                splits_name=args.splits_name,
                                model_fn=model_fn, 
                                metric=metric, 
                                tasks=tasks,                  
                                batch_size=batch_size,
                                vocab_data_path=args.vocab_data_path,
                                node_fdim=args.node_fdim,
                                edge_fdim=args.edge_fdim,
                                features_dim=args.feature_dim,
                                hidden_size=args.hidden_size,
                                functional_group_size=args.functional_group_size,
                                learning_rate=args.learning_rate, 
                                pretrained_model_path=pretrained_model_path, 
                                nb_epoch=epochs,
                                transformer_generators=transformers,
                                logger=logger)


if __name__ == "__main__":
    main()
