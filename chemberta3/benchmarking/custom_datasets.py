import deepchem as dc
from typing import List, Tuple
import pandas as pd


def load_nek(
    featurizer: dc.feat.Featurizer,
    tasks_wanted: List[str] = ["NEK2_ki_avg_value"],
    splitter=None,
) -> Tuple[List[str], Tuple[dc.data.Dataset, ...], List[dc.trans.Transformer], str]:
    """Load NEK dataset.

    The NEK dataset is a collection of datapoints related to the NEK kinases,
    including SMILES strings, ki, kd, inhibition, and ic50 values.

    NEK2_ki_avg_value contains the most non-nan data points. For other tasks,
    load the NEK .csv file directly from "s3://chemberta3/datasets/kinases/NEK/nek_mtss.csv".

    Mimics the loaders for molnet datasets.

    Parameters
    ----------
    featurizer: dc.feat.Featurizer
        Featurizer to use.
    tasks_wanted: List[str]
        Tasks to load. These should correspond to the columns in the dataframe.
    splitter: dc.splits.splitters.Splitter
        Splitter to use. This should be None, and is included only for compatibility.

    Returns
    -------
    tasks: List[str]
        List of tasks.
    datasets: Tuple[dc.data.Dataset, ...]
        Tuple of train, valid, test datasets.
    transformers: List[dc.trans.Transformer]
        List of transformers.

    """
    assert (
        splitter is None
    ), "Splitter arg only used for compatibility with other dataset loaders."
    nek_df = pd.read_csv(
        "s3://chemberta3/datasets/kinases/NEK/nek_mtss.csv", index_col=0
    )

    with dc.utils.UniversalNamedTemporaryFile(mode="w") as tmpfile:
        data_df = nek_df.dropna(subset=tasks_wanted)
        data_df.to_csv(tmpfile.name)
        loader = dc.data.CSVLoader(
            tasks_wanted, feature_field="raw_smiles", featurizer=featurizer
        )
        dc_dataset = loader.create_dataset(tmpfile.name)

    return [], [dc_dataset], []
