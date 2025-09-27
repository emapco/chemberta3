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

import os
import shutil
import warnings
from typing import cast

from datasets import Dataset as HFDataset
from datasets import load_dataset
from deepchem import data, feat, splits, trans
from deepchem.molnet.load_function.molnet_loader import (
    TransformerGenerator,
    _MolnetLoader,
)

ANTIMALARIAL_DATASET_ID = "Derify/antimalarial-classification"
ANTIMALARIAL_CSV = "antimalarial_classification.csv"
ANTIMALARIAL_TASKS = ["label"]

COCRYSTAL_DATASET_ID = "Derify/cocrystal-classification"
COCRYSTAL_CSV = "cocrystal_classification.csv"
COCRYSTAL_TASKS = ["label"]

COVID19_DATASET_ID = "Derify/covid-19-classification"
COVID19_CSV = "covid19_classification.csv"
COVID19_TASKS = ["label"]

ADME_TASKS = ["y"]
ADME_DATASETS = {
    "adme_microsom_stab_h": "adme_microsom_stab_h_cleaned.csv",
    "adme_microsom_stab_r": "adme_microsom_stab_r_cleaned.csv",
    "adme_permeability": "adme_permeability.csv",
    "adme_ppb_h": "adme_ppb_h.csv",
    "adme_ppb_r": "adme_ppb_r.csv",
    "adme_solubility": "adme_solubility.csv",
}
ASTRAZENECA_TASKS = ["y"]
ASTRAZENECA_DATASETS = {
    "astrazeneca_cl": "astrazeneca_CL.csv",
    "astrazeneca_logd74": "astrazeneca_LogD74.csv",
    "astrazeneca_ppb": "astrazeneca_PPB.csv",
    "astrazeneca_solubility": "astrazeneca_Solubility.csv",
}


class _AntimalarialLoader(_MolnetLoader):
    def create_dataset(self) -> data.Dataset:
        data_dir = self.data_dir
        if data_dir is None:
            raise ValueError("data_dir must be set before creating the antimalarial dataset.")

        dataset_file = os.path.join(data_dir, ANTIMALARIAL_CSV)
        if not os.path.exists(dataset_file):
            os.makedirs(data_dir, exist_ok=True)
            hf_dataset = cast(HFDataset, load_dataset(ANTIMALARIAL_DATASET_ID, split="train"))
            hf_dataset.to_csv(dataset_file)

        loader = data.CSVLoader(
            tasks=self.tasks,
            feature_field="smiles",
            featurizer=self.featurizer,  # type: ignore[arg-type]
        )
        return loader.create_dataset(dataset_file, shard_size=8192)


class _CocrystalLoader(_MolnetLoader):
    def create_dataset(self) -> data.Dataset:
        data_dir = self.data_dir
        if data_dir is None:
            raise ValueError("data_dir must be set before creating the cocrystal dataset.")

        dataset_file = os.path.join(data_dir, COCRYSTAL_CSV)
        if not os.path.exists(dataset_file):
            os.makedirs(data_dir, exist_ok=True)
            hf_dataset = cast(HFDataset, load_dataset(COCRYSTAL_DATASET_ID, split="train"))
            hf_dataset.to_csv(dataset_file)

        loader = data.CSVLoader(
            tasks=self.tasks,
            feature_field="smiles",
            featurizer=self.featurizer,  # type: ignore[arg-type]
        )
        return loader.create_dataset(dataset_file, shard_size=8192)


class _Covid19Loader(_MolnetLoader):
    def create_dataset(self) -> data.Dataset:
        data_dir = self.data_dir
        if data_dir is None:
            raise ValueError("data_dir must be set before creating the covid-19 dataset.")

        dataset_file = os.path.join(data_dir, COVID19_CSV)
        if not os.path.exists(dataset_file):
            os.makedirs(data_dir, exist_ok=True)
            hf_dataset = cast(HFDataset, load_dataset(COVID19_DATASET_ID, split="train"))
            hf_dataset.to_csv(dataset_file)

        loader = data.CSVLoader(
            tasks=self.tasks,
            feature_field="smiles",
            featurizer=self.featurizer,  # type: ignore[arg-type]
        )
        return loader.create_dataset(dataset_file, shard_size=8192)


class _AdmeLoader(_MolnetLoader):
    def create_dataset(self) -> data.Dataset:
        data_dir = self.data_dir
        if data_dir is None:
            raise ValueError("data_dir must be set before creating the ADME dataset.")

        dataset_name = self.args.get("dataset_name")
        if dataset_name is None:
            raise ValueError("dataset_name must be provided in args to load ADME dataset.")
        if dataset_name not in ADME_DATASETS:
            raise ValueError(f"Unknown ADME dataset: {dataset_name}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_dir = os.path.realpath(os.path.join(current_dir, "..", "datasets", "adme"))
        raw_dataset_file = os.path.join(raw_data_dir, ADME_DATASETS[dataset_name])
        if not os.path.exists(raw_dataset_file):
            raise FileNotFoundError(f"Raw ADME dataset file not found: {raw_dataset_file}")

        dataset_file = os.path.join(data_dir, ADME_DATASETS[dataset_name])
        if not os.path.exists(dataset_file):
            os.makedirs(data_dir, exist_ok=True)
            shutil.copy(raw_dataset_file, dataset_file)  # Copy the raw/preprocessed dataset to the data_dir

        loader = data.CSVLoader(
            tasks=self.tasks,
            feature_field="smiles",
            featurizer=self.featurizer,  # type: ignore[arg-type]
        )
        return loader.create_dataset(dataset_file, shard_size=8192)


class _AstrazenaLoader(_MolnetLoader):
    def create_dataset(self) -> data.Dataset:
        data_dir = self.data_dir
        if data_dir is None:
            raise ValueError("data_dir must be set before creating the Astrazeneca dataset.")

        dataset_name = self.args.get("dataset_name")
        if dataset_name is None:
            raise ValueError("dataset_name must be provided in args to load Astrazeneca dataset.")
        if dataset_name not in ASTRAZENECA_DATASETS:
            raise ValueError(f"Unknown Astrazeneca dataset: {dataset_name}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_dir = os.path.realpath(os.path.join(current_dir, "..", "datasets", "astrazeneca"))
        raw_dataset_file = os.path.join(raw_data_dir, ASTRAZENECA_DATASETS[dataset_name])
        if not os.path.exists(raw_dataset_file):
            raise FileNotFoundError(f"Raw Astrazeneca dataset file not found: {raw_dataset_file}")

        dataset_file = os.path.join(data_dir, ASTRAZENECA_DATASETS[dataset_name])
        if not os.path.exists(dataset_file):
            os.makedirs(data_dir, exist_ok=True)
            shutil.copy(raw_dataset_file, dataset_file)  # Copy the raw/preprocessed dataset to the data_dir

        loader = data.CSVLoader(
            tasks=self.tasks,
            feature_field="smiles",
            featurizer=self.featurizer,  # type: ignore[arg-type]
        )
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_antimalarial(
    featurizer: feat.Featurizer | str = "ECFP",
    splitter: splits.Splitter | str | None = "scaffold",
    transformers: list[TransformerGenerator | str] = None,  # type: ignore
    reload: bool = True,
    data_dir: str | None = None,
    save_dir: str | None = None,
    **kwargs,
) -> tuple[list[str], tuple[data.Dataset, ...], list[trans.Transformer]]:
    """Load antimalarial classification dataset.

    Parameters
    ----------
    featurizer: Featurizer or str
            The featurizer to use for processing the data.
    splitter: Splitter or str
            The splitter to use for splitting the data into training, validation,
            and test sets.
    transformers: list of TransformerGenerators or strings
            The transformers to apply to the data.
    reload: bool
            Whether to reload a cached version of the dataset if available.
    data_dir: str
            Directory to save the raw data in.
    save_dir: str
            Directory to save processed datasets in.
    """
    if transformers is None:
        transformers = ["balancing"]
    if splitter == "scaffold":
        # Use random split instead of scaffold for antimalarial dataset
        splitter = "random"
        warnings.warn(
            "Using random splitter instead of scaffold for antimalarial dataset. "
            "Scaffold splitter produces an invalid test set with no positive examples.",
            stacklevel=2,
        )
    loader = _AntimalarialLoader(
        featurizer=featurizer,
        splitter=splitter,
        transformer_generators=transformers,
        tasks=ANTIMALARIAL_TASKS,
        data_dir=data_dir,
        save_dir=save_dir,
        **kwargs,
    )
    return loader.load_dataset("antimalarial", reload)


def load_cocrystal(
    featurizer: feat.Featurizer | str = "ECFP",
    splitter: splits.Splitter | str | None = "scaffold",
    transformers: list[TransformerGenerator | str] = None,  # type: ignore
    reload: bool = True,
    data_dir: str | None = None,
    save_dir: str | None = None,
    **kwargs,
) -> tuple[list[str], tuple[data.Dataset, ...], list[trans.Transformer]]:
    """Load cocrystal classification dataset.

    Parameters
    ----------
    featurizer: Featurizer or str
        The featurizer to use for processing the data.
    splitter: Splitter or str
        The splitter to use for splitting the data into training, validation,
        and test sets.
    transformers: list of TransformerGenerators or strings
        The transformers to apply to the data.
    reload: bool
        Whether to reload a cached version of the dataset if available.
    data_dir: str
        Directory to save the raw data in.
    save_dir: str
        Directory to save processed datasets in.
    """
    if transformers is None:
        transformers = ["balancing"]
    loader = _CocrystalLoader(
        featurizer=featurizer,
        splitter=splitter,
        transformer_generators=transformers,
        tasks=COCRYSTAL_TASKS,
        data_dir=data_dir,
        save_dir=save_dir,
        **kwargs,
    )
    return loader.load_dataset("cocrystal", reload)


def load_covid19(
    featurizer: feat.Featurizer | str = "ECFP",
    splitter: splits.Splitter | str | None = "scaffold",
    transformers: list[TransformerGenerator | str] = None,  # type: ignore
    reload: bool = True,
    data_dir: str | None = None,
    save_dir: str | None = None,
    **kwargs,
) -> tuple[list[str], tuple[data.Dataset, ...], list[trans.Transformer]]:
    """Load covid-19 classification dataset.

    Parameters
    ----------
    featurizer: Featurizer or str
        The featurizer to use for processing the data.
    splitter: Splitter or str
        The splitter to use for splitting the data into training, validation,
        and test sets.
    transformers: list of TransformerGenerators or strings
        The transformers to apply to the data.
    reload: bool
        Whether to reload a cached version of the dataset if available.
    data_dir: str
        Directory to save the raw data in.
    save_dir: str
        Directory to save processed datasets in.
    """
    if transformers is None:
        transformers = ["balancing"]
    loader = _Covid19Loader(
        featurizer=featurizer,
        splitter=splitter,
        transformer_generators=transformers,
        tasks=COVID19_TASKS,
        data_dir=data_dir,
        save_dir=save_dir,
        **kwargs,
    )
    return loader.load_dataset("covid19", reload)


def load_adme_dataset(
    featurizer: feat.Featurizer | str = "ECFP",
    splitter: splits.Splitter | str | None = "scaffold",
    transformers: list[TransformerGenerator | str] = None,  # type: ignore
    reload: bool = True,
    data_dir: str | None = None,
    save_dir: str | None = None,
    dataset_name: str = "adme_microsom_stab_h",
    **kwargs,
) -> tuple[list[str], tuple[data.Dataset, ...], list[trans.Transformer]]:
    """Load covid-19 classification dataset.

    Parameters
    ----------
    featurizer: Featurizer or str
        The featurizer to use for processing the data.
    splitter: Splitter or str
        The splitter to use for splitting the data into training, validation,
        and test sets.
    transformers: list of TransformerGenerators or strings
        The transformers to apply to the data.
    reload: bool
        Whether to reload a cached version of the dataset if available.
    data_dir: str
        Directory to save the raw data in.
    save_dir: str
        Directory to save processed datasets in.
    """
    if transformers is None:
        transformers = ["balancing"]
    loader = _AdmeLoader(
        featurizer=featurizer,
        splitter=splitter,
        transformer_generators=transformers,
        tasks=ADME_TASKS,
        data_dir=data_dir,
        save_dir=save_dir,
        dataset_name=dataset_name,
        **kwargs,
    )
    return loader.load_dataset(dataset_name, reload)


def load_astrazeneca_dataset(
    featurizer: feat.Featurizer | str = "ECFP",
    splitter: splits.Splitter | str | None = "scaffold",
    transformers: list[TransformerGenerator | str] = None,  # type: ignore
    reload: bool = True,
    data_dir: str | None = None,
    save_dir: str | None = None,
    dataset_name: str = "adme_microsom_stab_h",
    **kwargs,
) -> tuple[list[str], tuple[data.Dataset, ...], list[trans.Transformer]]:
    """Load covid-19 classification dataset.

    Parameters
    ----------
    featurizer: Featurizer or str
        The featurizer to use for processing the data.
    splitter: Splitter or str
        The splitter to use for splitting the data into training, validation,
        and test sets.
    transformers: list of TransformerGenerators or strings
        The transformers to apply to the data.
    reload: bool
        Whether to reload a cached version of the dataset if available.
    data_dir: str
        Directory to save the raw data in.
    save_dir: str
        Directory to save processed datasets in.
    """
    if transformers is None:
        transformers = ["balancing"]
    loader = _AstrazenaLoader(
        featurizer=featurizer,
        splitter=splitter,
        transformer_generators=transformers,
        tasks=ADME_TASKS,
        data_dir=data_dir,
        save_dir=save_dir,
        dataset_name=dataset_name,
        **kwargs,
    )
    return loader.load_dataset(dataset_name, reload)
