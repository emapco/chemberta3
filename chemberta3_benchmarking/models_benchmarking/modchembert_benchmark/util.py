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

import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import torch
from deepchem.models.optimizers import Optimizer
from optimi import StableAdamW


@dataclass
class DatasetHyperparameters:
    batch_size: int
    epochs: int
    learning_rate: float
    classifier_pooling: str
    classifier_pooling_last_k: int
    classifier_pooling_attention_dropout: float
    classifier_dropout: float
    embedding_dropout: float
    use_normalized_weight_decay: bool
    weight_decay: float = 0.0
    optimizer: Literal["adam", "stable_adamw"] = "adam"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"


class WrappedStableAdamW(Optimizer):
    """An algorithm for optimizing a model.

    This is an abstract class.  Subclasses represent specific optimization algorithms.
    """

    def __init__(self, learning_rate: float, weight_decay: float):
        """This constructor should only be called by subclasses.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        """
        super().__init__(learning_rate=learning_rate)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _create_pytorch_optimizer(self, params):
        """Construct a PyTorch optimizer.

        Parameters
        ----------
        params: Iterable
            the model parameters to optimize

        Returns
        -------
        a new PyTorch optimizer implementing the algorithm
        """
        return StableAdamW(
            params=params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            # Workaround: restoring optimizer state can raise KeyError 'mean_square' when Triton is enabled.
            triton=False,
        )


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


def setup_logging(
    pretrained_model_path: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    model_cfg: dict,
    task: Literal["classification", "regression"],
) -> logging.Logger:
    """Set up logging for the experiment.

    Parameters
    ----------
    pretrained_model_path: str
        Path to the pretrained ModChemBERT model.
    dataset: str
        Name of the dataset being used.
    epochs: int
        Number of epochs for training.
    batch_size: int
        Batch size used for training.
    model_cfg: dict
        Configuration dictionary for the ModChemBERT model for overriding the pretrained model config.


    Returns
    -------
    logger : logging.Logger
        Configured logger for the experiment.
    """
    # Create a directory for logs if it doesn't exist
    parsed_model_name = os.path.basename(pretrained_model_path.rstrip("/"))
    log_dir = (
        f"logs_modchembert_{task}_{parsed_model_name}_"
        f"{model_cfg['classifier_pooling']}_pooling_last_k{model_cfg['classifier_pooling_last_k']}"
    )
    log_dir = os.path.join("outputs", log_dir)
    os.makedirs(log_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        log_dir, f"modchembert_deepchem_splits_run_{dataset}_epochs{epochs}_batch_size{batch_size}_{datetime_str}.log"
    )

    logger = logging.getLogger(f"logs_modchembert_{dataset}_epochs{epochs}_batch_size{batch_size}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger
