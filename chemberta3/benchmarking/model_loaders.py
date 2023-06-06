import deepchem as dc
from deepchem.models.torch_models import InfoGraphModel
from deepchem.models.sklearn_models import SklearnModel
from typing import List, Optional


def load_infograph(num_feat: int, edge_dim: int, metrics: List[dc.metrics.Metric], checkpoint_path: Optional[str] = None):
    """Load an InfoGraph model.
    Parameters
    ----------
    num_feat: int
        Number of atom features
    edge_dim: int
        Number of edge features
    metrics: List[dc.metrics.Metric]
        List of metrics to use.
    checkpoint_path: str
        Path to model checkpoint

    Returns
    -------
    model: dc.models.torch_models.modular.ModularTorchModel
        Loaded model.
    """
    # NOTE: cannot pass in `self.loss` because of how InfoGraphModel is constructed
    # TODO: fix this (only regression currently supported)
    model = InfoGraphModel(
        num_feat,
        edge_dim,
        64,
        use_unsup_loss=False,
        separate_encoder=True,
        metrics=metrics,
    )
    if checkpoint_path is not None:
        model.load_pretrained_components(checkpoint=checkpoint_path)
    return model 
