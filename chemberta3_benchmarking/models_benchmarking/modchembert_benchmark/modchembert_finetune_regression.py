# Copyright 2025 Emmanuel Cortes, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import shutil
from collections.abc import Callable
from datetime import datetime

import hydra
import numpy as np
import torch
from deepchem.data import DiskDataset
from deepchem.metrics import Metric, mae_score, mean_absolute_error, mean_squared_error, rms_score
from deepchem.models.optimizers import Adam
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator
from deepchem.trans import (
    BalancingTransformer,
    ClippingTransformer,
    LogTransformer,
    MinMaxTransformer,
    NormalizationTransformer,
)
from omegaconf import DictConfig, OmegaConf
from util import DatasetHyperparameters, WrappedStableAdamW, set_seed, setup_logging

from modchembert.models.modchembert import ModChemBert

transformers_mapping = {
    "balancing": TransformerGenerator(BalancingTransformer),
    "normalization": TransformerGenerator(NormalizationTransformer, transform_y=True),
    "minmax": TransformerGenerator(MinMaxTransformer, transform_y=True),
    "clipping": TransformerGenerator(ClippingTransformer, transform_y=True),
    "log": TransformerGenerator(LogTransformer, transform_y=True),
}


def transform_splits(
    train_dataset: DiskDataset,
    valid_dataset: DiskDataset,
    test_dataset: DiskDataset,
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
    tasks: list,
    model_dir: str,
    pretrained_model_path: str,
    model_config: dict,
    dataset_hparams: DatasetHyperparameters,
) -> ModChemBert:
    """Create a ModChemBERT model for regression tasks using dataset-specific hyperparameters."""
    model_config["classifier_pooling"] = dataset_hparams.classifier_pooling
    model_config["classifier_pooling_last_k"] = dataset_hparams.classifier_pooling_last_k
    model_config["classifier_pooling_attention_dropout"] = dataset_hparams.classifier_pooling_attention_dropout
    model_config["classifier_dropout"] = dataset_hparams.classifier_dropout
    model_config["embedding_dropout"] = dataset_hparams.embedding_dropout
    if dataset_hparams.optimizer == "adam":
        Optimizer = Adam
    elif dataset_hparams.optimizer == "stable_adamw":
        Optimizer = WrappedStableAdamW
    else:
        raise ValueError(f"Unsupported optimizer: {dataset_hparams.optimizer}")
    finetune_model = ModChemBert(
        task="regression",
        config=model_config,
        optimizer=Optimizer(
            learning_rate=dataset_hparams.learning_rate,
            weight_decay=dataset_hparams.weight_decay,
        ),
        batch_size=dataset_hparams.batch_size,
        n_tasks=len(tasks),
        model_dir=model_dir,
        tokenizer_path=pretrained_model_path,
        pretrained_model=pretrained_model_path,
        dtype=dataset_hparams.torch_dtype,
    )
    return finetune_model


def run_deepchem_experiment(
    run_id: int,
    model_fn: Callable,
    train_dataset: DiskDataset,
    valid_dataset: DiskDataset,
    test_dataset: DiskDataset,
    metric: Metric,
    dataset: str,
    tasks: list,
    model_dir: str,
    pretrained_model_path: str,
    model_config: dict,
    dataset_hparams: DatasetHyperparameters,
    transformer_generators: list,
    logger: logging.Logger | None = None,
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
    pretrained_model_path: str
        Path to the pretrained ModChemBERT model.
    model_config: dict
        Configuration dictionary for the ModChemBERT model for overriding the pretrained model config.
    dataset_hparams: DatasetHyperparameters
        Per-dataset training hyperparameters.
    transformer_generators: list[Union[str, TransformerGenerator]]
        A list of transformer generator objects or string names representing them. If a string is
        passed, it is resolved using the `transformers_mapping` dictionary. Each generator should
        implement a `create_transformer` method.
    logger: logging.Logger
        Logger for the experiment.

    Returns
    -------
    test_score: float
        Test score of the model.
    """
    set_seed(run_id)
    if dataset_hparams.use_normalized_weight_decay is True:
        dataset_hparams.weight_decay = 0.05 * math.sqrt(
            dataset_hparams.batch_size / (len(train_dataset) * dataset_hparams.epochs)
        )
    model = model_fn(
        tasks=tasks,
        model_dir=model_dir,
        pretrained_model_path=pretrained_model_path,
        model_config=model_config,
        dataset_hparams=dataset_hparams,
    )
    best_score = np.inf

    # Get current datetime and format it
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    epochs = dataset_hparams.epochs
    modchembert_model_dir = os.path.join("training_output", "chemberta3-modchembert-ft-models", "checkpoints", dataset)
    best_model_dir = os.path.join(modchembert_model_dir, f"best_model_epochs{epochs}_run{run_id}_{current_datetime}")
    os.makedirs(modchembert_model_dir, exist_ok=True)

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

        if logger:
            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss:.4f} | Val {metric.name}: {val_score:.4f}")

        if epoch % 5 == 0 and epoch > 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if val_score < best_score:
            best_score = val_score

            # Save current checkpoint as best model
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
            if logger:
                logger.info(f"Global step of best model: {model._global_step}")
            shutil.copytree(model.model_dir, best_model_dir)

            if logger:
                logger.info(f"Best model saved at epoch {epoch + 1} with val {metric.name}: {val_score:.4f}")

    # Load best checkpoint before evaluating on test set
    model.restore(model_dir=best_model_dir)
    test_scores = model.evaluate(dataset=test_dataset, metrics=[metric], transformers=transformers)
    test_score = test_scores[metric.name]
    if logger:
        logger.info(f"Test {metric.name}: {test_score:.4f}")

    return test_score


def triplicate_benchmark_dc(
    dataset: str,
    model_fn: Callable,
    metric: Metric,
    tasks: list,
    pretrained_model_path: str,
    model_config: dict,
    dataset_hparams: DatasetHyperparameters,
    transformer_generators: list,
    logger: logging.Logger | None = None,
) -> tuple[float, float]:
    """Run a triplicate benchmark for the given dataset.

    Parameters
    ----------
    dataset: str
        Name of the dataset being used.
    model_fn: function
        Function to create the model.
    metric: dc.metrics.Metric
        Metric to evaluate the model.
    tasks: list
        list of tasks for the model.
    pretrained_model_path: str
        Path to the pretrained ModChemBERT model.
    model_config: dict
        Configuration dictionary for the ModChemBERT model for overriding the pretrained model config.
    dataset_hparams: DatasetHyperparameters
        Per-dataset training hyperparameters (batch_size, learning_rate, epochs, classifier_pooling).
    transformer_generators: list[Union[str, TransformerGenerator]]
        A list of transformer generator objects or string names representing them. If a string is
        passed, it is resolved using the `transformers_mapping` dictionary. Each generator should
        implement a `create_transformer` method.
    logger: logging.Logger
        Logger for the experiment.

    Returns
    -------
    avg_score: float
        Average score of the triplicate runs.
    std_score: float
        Standard deviation of the triplicate runs.
    """
    scores = []
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(
        os.path.join(current_file_path, f"../../data/featurized_datasets/deepchem_splits/dummy_featurized/{dataset}")
    )
    train_dataset = DiskDataset(os.path.join(dataset_path, "train"))
    valid_dataset = DiskDataset(os.path.join(dataset_path, "valid"))
    test_dataset = DiskDataset(os.path.join(dataset_path, "test"))

    for run_id in range(3):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if logger:
            logger.info(f"Starting triplicate run {run_id + 1} for dataset {dataset} at {current_datetime}")
        modchembert_model_dir = os.path.join("training_output", "chemberta3-modchembert-ft-models")
        os.makedirs(modchembert_model_dir, exist_ok=True)
        model_dir = os.path.join(modchembert_model_dir, f"{dataset}_{run_id}_{current_datetime}")
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
            pretrained_model_path=pretrained_model_path,
            model_config=model_config,
            dataset_hparams=dataset_hparams,
            transformer_generators=transformer_generators,
            logger=logger,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        scores.append(test_score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    if logger:
        logger.info(f"Final Triplicate Test Results — Avg {metric.name}: {avg_score:.4f}, Std Dev: {std_score:.4f}")
    return float(avg_score), float(std_score)


@hydra.main(config_path="../../../../conf/chemberta3", config_name="benchmark-regression", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main function to run the ModChemBERT benchmark using Hydra config."""

    datasets = list(cfg.datasets)
    if len(datasets) == 0:
        raise ValueError("The list of datasets is empty. Please provide at least one dataset.")

    pretrained_model_path = cfg.pretrained_model_path
    if pretrained_model_path is None:
        raise ValueError("Please provide a path to the pretrained model.")

    task_dict = {
        "delaney": ["measured_log_solubility_in_mols_per_litre"],
        "freesolv": ["y"],
        "lipo": ["exp"],
        "clearance": ["target"],
        "bace_regression": ["pIC50"],
        "adme_microsom_stab_h": ["y"],
        "adme_microsom_stab_r": ["y"],
        "adme_permeability": ["y"],
        "adme_ppb_h": ["y"],
        "adme_ppb_r": ["y"],
        "adme_solubility": ["y"],
        "astrazeneca_cl": ["y"],
        "astrazeneca_logd74": ["y"],
        "astrazeneca_ppb": ["y"],
        "astrazeneca_solubility": ["y"],
    }

    transformer_generators = {
        "delaney": ["normalization"],
        "freesolv": ["normalization"],
        "lipo": ["normalization"],
        "clearance": ["log"],
        "bace_regression": ["normalization"],
        "adme_microsom_stab_h": ["normalization"],
        "adme_microsom_stab_r": ["normalization"],
        "adme_permeability": ["normalization"],
        "adme_ppb_h": ["normalization"],
        "adme_ppb_r": ["normalization"],
        "adme_solubility": ["normalization"],
        "astrazeneca_cl": ["normalization"],
        "astrazeneca_logd74": ["normalization"],
        "astrazeneca_ppb": ["normalization"],
        "astrazeneca_solubility": ["normalization"],
    }

    # Metric for regression tasks
    metric_name = str(cfg.metric)
    if metric_name == "rms_score":
        metric = Metric(rms_score)
    elif metric_name == "mean_squared_error":
        metric = Metric(mean_squared_error)
    elif metric_name == "mean_absolute_error":
        metric = Metric(mean_absolute_error)
    elif metric_name == "mae_score":
        metric = Metric(mae_score)
    else:
        raise ValueError(f"Unsupported metric: {cfg.metric}")

    model_cfg = OmegaConf.to_container(cfg.modchembert_config, resolve=True)
    assert isinstance(model_cfg, dict), "modchembert_config could not be converted to dict"

    for dataset in datasets:
        if dataset not in task_dict:
            raise ValueError(f"Dataset {dataset} not found in task_dict.")
        if dataset not in cfg.dataset_hyperparameters:
            raise ValueError(f"Missing dataset_hyperparameters for dataset: {dataset}")
        ds_hp_dict = OmegaConf.to_container(cfg.dataset_hyperparameters[dataset], resolve=True)
        assert isinstance(ds_hp_dict, dict), "dataset_hyperparameters must convert to dict"
        dataset_hparams = DatasetHyperparameters(
            batch_size=int(ds_hp_dict["batch_size"]),
            epochs=int(ds_hp_dict["epochs"]),
            learning_rate=float(ds_hp_dict["learning_rate"]),
            classifier_pooling=ds_hp_dict.get("classifier_pooling", model_cfg.get("classifier_pooling", "sum_mean")),
            classifier_pooling_last_k=int(
                ds_hp_dict.get("classifier_pooling_last_k", model_cfg.get("classifier_pooling_last_k", 3))
            ),
            classifier_pooling_attention_dropout=float(
                ds_hp_dict.get(
                    "classifier_pooling_attention_dropout", model_cfg.get("classifier_pooling_attention_dropout", 0.1)
                )
            ),
            classifier_dropout=float(ds_hp_dict.get("classifier_dropout", model_cfg.get("classifier_dropout", 0.0))),
            embedding_dropout=float(ds_hp_dict.get("embedding_dropout", model_cfg.get("embedding_dropout", 0.0))),
            use_normalized_weight_decay=bool(
                ds_hp_dict.get("use_normalized_weight_decay", cfg.get("use_normalized_weight_decay", False))
            ),
            weight_decay=float(ds_hp_dict.get("weight_decay", cfg.get("weight_decay", 0.0))),
            optimizer=ds_hp_dict.get("optimizer", cfg.get("optimizer", "adam")),
            torch_dtype=ds_hp_dict.get("torch_dtype", cfg.get("torch_dtype", "bfloat16")),
        )

        transformers = transformer_generators[dataset] if cfg.transform is True else []
        tasks = task_dict[dataset]

        logger = setup_logging(
            pretrained_model_path=pretrained_model_path,
            dataset=dataset,
            epochs=dataset_hparams.epochs,
            batch_size=dataset_hparams.batch_size,
            model_cfg={
                **model_cfg,
                "classifier_pooling": dataset_hparams.classifier_pooling or model_cfg.get("classifier_pooling"),
                "classifier_pooling_last_k": dataset_hparams.classifier_pooling_last_k
                or model_cfg.get("classifier_pooling_last_k"),
            },
            task="regression",
        )
        logger.info(f"Running benchmark for dataset: {dataset}")
        logger.info(
            f"dataset: {dataset}, tasks: {tasks}, epochs: {dataset_hparams.epochs}, "
            f"learning rate: {dataset_hparams.learning_rate}, transform: {cfg.transform}"
        )

        triplicate_benchmark_dc(
            dataset=dataset,
            model_fn=model_fn,
            metric=metric,
            tasks=tasks,
            pretrained_model_path=pretrained_model_path,
            model_config=model_cfg,
            dataset_hparams=dataset_hparams,
            transformer_generators=transformers,
            logger=logger,
        )


if __name__ == "__main__":
    main()
