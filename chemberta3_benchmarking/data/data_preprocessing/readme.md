# ChemBERTa3 Dataset Featurization Pipeline

This utility streamlines the process of preparing featurized datasets for machine learning models. 
It supports:

- Generating train/valid/test splits of moleculenet datasets using **DeepChem**
- Featurizing datasets with multiple **DeepChem featurizers**
- Processing and featurizing pre-split datasets from **external sources** (e.g., **Molformer**)

## Usage

1. Generate DeepChem Scaffold Splits (Optional)

Use this if your --split_type is deepchem and you want to generate train, valid, test splits.

```
        python3 prepare_data.py\
            --split_type 'deepchem' \
            --datasets 'delaney' \
            --featurizers 'ecfp' \
            --data_dir ./../datasets/deepchem_splits \
            --feat_dir ./../featurized_datasets/deepchem_splits \
```

Note - Current recommendation from deepchem for freesolv is Random splitting.

2. Featurize Any Dataset (DeepChem or Molformer)

If you already have split CSVs (e.g., from Molformer), set --split_type molformer (or skip it altogether):

```
        python3 prepare_data.py\
            --split_type 'molformer' \
            --datasets 'delaney' \
            --featurizers 'ecfp' \
            --data_dir ./../datasets/molformer_splits \
            --feat_dir ./../featurized_datasets/molformer_splits \
```

## Command-Line Arguments

Use the following arguments to control dataset splitting, cleaning, and featurization behavior.

| Argument        | Description                                         |
| --------------- | --------------------------------------------------- |
| `--split_type`  | `deepchem` or `molformer` (default: `molformer`)    |
| `--datasets`    | Comma-separated list of dataset names               |
| `--featurizers` | Comma-separated list of featurizer keys (see below) |
| `--data_dir`    | Path to input/output CSVs                           |
| `--feat_dir`    | Path to store featurized datasets                   |

## Supported DeepChem featurizers

This pipeline supports the following DeepChem featurizers:

| Featurizer Key    | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| `dmpnn`           | Directed Message Passing Neural Network                                |
| `dummy`           | No featurization, returns raw SMILES                                   |
| `grover`          | Graph-based features using Grover, combined with circular fingerprints.|
| `ecfp`            | Extended Connectivity Fingerprint (ECFP)                               |
| `molgraphconv`    | GraphConv features using edge info                                     |
| `rdkit_conformer` | 3D conformer-based features generated via RDKit.                       |

## Data cleaning

This pipeline filters out SMILES longer than ~200 characters, following the MoLFormer paper's recommendation. ([MoLFormer paper](Large-scale chemical language representations capture molecular structure and properties)(https://doi.org/10.1038/s42256-022-00580-7)).

## Output Structure

```
./deepchem_splits/
└── dataset/
    ├── train.csv
    ├── valid.csv
    └── test.csv

./featurized/
└── dmpnn_featurized/
    └── dataset/
        ├── train/
        ├── valid/
        └── test/
```

## Logs

Logs are saved in:

```
./deepchem_splits/deepchem_split_log_<timestamp>.log

./featurized/featurization_log_<timestamp>.log
```





