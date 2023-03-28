import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models.torch_models import InfoGraphModel


class BenchmarkingDatasetLoader:
    def __init__(self) -> None:
        self.dataset_mapping = {
            "bace_classification": {
                "loader": dc.molnet.load_bace_classification,
                "output_type": "classification",
            },
            "bace_regression": {
                "loader": dc.molnet.load_bace_regression,
                "output_type": "regression",
            },
            "bbbp": {
                "loader": dc.molnet.load_bbbp,
                "output_type": "classification",
            },
            "clintox": {
                "loader": dc.molnet.load_clintox,
                "output_type": "classification",
                "tasks_wanted": ["CT_TOX"],
            },
            "delaney": {
                "loader": dc.molnet.load_delaney,
                "output_type": "regression",
            },
            "hiv": {
                "loader": dc.molnet.load_hiv,
                "output_type": "classification",
            },
            "muv": {"loader": dc.molnet.load_muv, "output_type": "classification"},
            "pcba": {"loader": dc.molnet.load_pcba, "output_type": "classification"},
            "qm9": {
                "dataset_type": "regression",
                "load_fn": dc.molnet.load_qm9,
            },
            "tox21": {
                "loader": dc.molnet.load_tox21,
                "output_type": "classification",
                "tasks_wanted": ["SR-p53"],
            },
            "nek": {
                "loader": self._load_nek,
                "output_type": "regression",
                "tasks_wanted": ["NEK2_ki_avg_value"],
            },
        }

    @property
    def dataset_names(self):
        return list(self.dataset_mapping.keys())

    def _load_nek(
        self,
        featurizer: dc.feat.Featurizer,
        tasks_wanted: List[str] = ["NEK2_ki_avg_value"],
        splitter=None,
        **kwargs,
    ):
        assert (
            splitter is None
        ), "Splitter arg only used for compatibility with other dataset loaders."
        nek_df = pd.read_csv(
            "~/Playground/data_download/kinases/NEK/nek_mtss.csv", index_col=0
        )

        with dc.utils.UniversalNamedTemporaryFile(mode="w") as tmpfile:
            data_df = nek_df.dropna(subset=tasks_wanted)
            data_df.to_csv(tmpfile.name)
            loader = dc.data.CSVLoader(
                tasks_wanted, feature_field="raw_smiles", featurizer=featurizer
            )
            dc_dataset = loader.create_dataset(tmpfile.name)

        return [], [dc_dataset], []

    def load_dataset(self, dataset_name: str, featurizer: dc.feat.Featurizer):
        if dataset_name not in self.dataset_mapping:
            raise ValueError(f"Dataset {dataset_name} not found in dataset mapping.")

        dataset_loader = self.dataset_mapping[dataset_name]["loader"]
        output_type = self.dataset_mapping[dataset_name]["output_type"]
        tasks, datasets, transformers = dataset_loader(
            featurizer=featurizer, splitter=None
        )
        return tasks, datasets, transformers, output_type


class BenchmarkingModelLoader:
    def __init__(
        self, loss: dc.models.losses.Loss, metrics: List[dc.metrics.Metric]
    ) -> None:
        self.loss = loss
        self.metrics = metrics
        self.model_mapping = {
            "infograph": self._load_infograph,
        }

    def _load_infograph(self, num_feat: int, edge_dim: int):
        # NOTE: cannot pass in `self.loss` because of how InfoGraphModel is constructed
        # TODO: fix this (only regression currently supported)
        model = InfoGraphModel(
            num_feat,
            edge_dim,
            64,
            use_unsup_loss=False,
            separate_encoder=True,
            metrics=self.metrics,
        )
        return model

    def load_model(
        self,
        model_name: str,
        checkpoint_path: str = None,
        model_loading_kwargs: Dict = {},
    ):
        if model_name not in self.model_mapping:
            raise ValueError(f"Model {model_name} not found in model mapping.")

        model_loader = self.model_mapping[model_name]
        model = model_loader(**model_loading_kwargs)
        model.load_pretrained_components(checkpoint=checkpoint_path)
        return model


def get_infograph_loading_kwargs(dataset):
    num_feat = max([dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max([dataset.X[i].num_edge_features for i in range(len(dataset))])
    return {"num_feat": num_feat, "edge_dim": edge_dim}


@dataclass
class EarlyStopper:
    patience: int = 5
    min_delta: float = 0.0
    min_loss: float = np.inf
    best_epoch: int = 0

    def __call__(self, loss: float, epoch: int) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.best_epoch = epoch
            return False
        elif loss - self.min_loss > self.min_delta:
            if epoch - self.best_epoch > self.patience:
                return True
        return False


class BenchmarkingFeaturizerLoader:
    def __init__(self) -> None:
        self.featurizer_mapping = {
            "molgraphconv": MolGraphConvFeaturizer(use_edges=True),
        }

    def load_featurizer(self, featurizer_name: str):
        if featurizer_name not in self.featurizer_mapping:
            raise ValueError(
                f"Featurizer {featurizer_name} not found in featurizer mapping."
            )

        featurizer = self.featurizer_mapping[featurizer_name]
        return featurizer


def train(args):
    dataset_loader = BenchmarkingDatasetLoader()
    featurizer_loader = BenchmarkingFeaturizerLoader()

    splitter = dc.splits.ScaffoldSplitter()
    featurizer = featurizer_loader.load_featurizer(args.featurizer_name)

    tasks, datasets, transformers, output_type = dataset_loader.load_dataset(
        args.dataset_name, featurizer
    )
    unsplit_dataset = datasets[0]
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        unsplit_dataset
    )

    metrics = (
        [dc.metrics.Metric(dc.metrics.pearson_r2_score)]
        if output_type == "regression"
        else [dc.metrics.Metric(dc.metrics.auc, mode="classification")]
    )
    loss = (
        dc.models.losses.L2Loss()
        if output_type == "regression"
        else dc.models.losses.BinaryCrossEntropy()
    )

    model_loader = BenchmarkingModelLoader(loss=loss, metrics=metrics)
    model_loading_kwargs = {}
    if args.model_name == "infograph":
        model_loading_kwargs = get_infograph_loading_kwargs(train_dataset)

    model = model_loader.load_model(
        args.model_name, args.checkpoint, model_loading_kwargs
    )

    early_stopper = EarlyStopper()

    for epoch in range(50):
        training_loss_value = model.fit(train_dataset, nb_epoch=1)
        eval_preds = model.predict(valid_dataset)
        eval_loss_fn = loss._create_pytorch_loss()
        eval_loss = torch.sum(
            eval_loss_fn(torch.Tensor(eval_preds), torch.Tensor(valid_dataset.y))
        ).item()

        eval_metrics = model.evaluate(
            valid_dataset,
            metrics=metrics,
        )
        print(
            f"Epoch {epoch} training loss: {training_loss_value}; validation loss: {eval_loss}; validation metrics: {eval_metrics}"
        )
        if early_stopper(eval_loss, epoch):
            break

    # compute test metrics
    test_metrics = model.evaluate(test_dataset, metrics=metrics)
    test_metrics_df = pd.DataFrame.from_dict(
        {k: np.array(v) for k, v in test_metrics.items()}, orient="index"
    )
    print(f"Test metrics: {test_metrics_df}")
    test_metrics_df.to_csv(
        f"{args.model_name}_{args.dataset_name}_test_metrics.csv",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="infograph")
    argparser.add_argument("--featurizer_name", type=str, default="molgraphconv")
    argparser.add_argument("--dataset_name", type=str, default="nek")
    argparser.add_argument("--checkpoint", type=str, default=None)
    args = argparser.parse_args()
    train(args)
