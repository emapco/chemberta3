# Chemberta3

**Chemberta3** is an open-source training framework designed to train and fine-tune large-scale chemical foundation models. We explore the potential of multiple model architectures, by evaluating their performance across various molecular datasets from the MoleculeNet suite.

1. [Getting Started](#getting-started)
    1. [Pretrained Models and training logs](#pretrained-models-and-training-logs)
    2. [Replicating Conda Environment](#replicating-conda-environment)
2. [Data](#data)
    1. [Pretraining Datasets](#pretraining-datasets)
    2. [Finetuning Datasets](#finetuning-datasets)
3. [Pretraining](#pretraining)
4. [Finetuning](#finetuning)
5. [Citations](#citatiobs)


## Getting Started

**This Code and Environment have been tested on Nvidia T4 GPUs**

#### Pretrained Models and training logs
We are providing checkpoints of the MoLFomer model pre-trained on a dataset of 1.1B molecules. This dataset combines 100% of Zinc and 100% of PubChem molecules used for training. We are also providing
best checkpoints for grover, chemberta, infograph, infomax3d pretrained on 250K ZINC molecules.

Extract `Pretrained MoLFormer.zip` containing the pretrained models and associated training logs to the `data/` directory.
The hierarchy should look like the following:

```
data/
├── finetune_datasets
├── pretrained_model_checkpoints
│   ├── pretrained_chemberta
│   ├── pretrained_grover
│   ├── pretrained_infomax3d
│   ├── pretrained_infograph
│   ├── pretrained_molformer
│   ├── pretrained_molformer_llnl
│   └── hparams.yaml
```

#### Replicating Conda Environment

## Data

MolFormer splits are available at [https://ibm.box.com/v/MoLFormer-data](https://ibm.box.com/v/MoLFormer-data)

### Pretraining Datasets


### Finetuning Datasets
Just as with the pretraining data the code expects the featurized finetuning datasets to be in the following hierarchy. These datasets were provided in the `finetune_datasets.zip`, they can also be
generated using the respective featurization scripts. 

```
data/
|   ├── featurized_datasets
|   |   ├── deepchem splits
|   |   |   ├── dmpnn_featurized
|   │   │   ├── dummy_featurized
|   │   │   ├── ecfp_featurized
|   │   │   ├── molgraphconv_featurized
|   │   │   ├── rdkit_conformer_featurized
|   |   ├── molformer splits
|   |   |   ├── dmpnn_featurized
|   │   │   ├── dummy_featurized
|   │   │   ├── ecfp_featurized
|   │   │   ├── molgraphconv_featurized
|   │   │   ├── rdkit_conformer_featurized
|   ├── finetune_datasets
|   |   ├── deepchem splits
|   |   |   ├── bace
|   |   |   |   ├── test.csv
│   |   |   |   ├── train.csv
│   |   |   |   └── valid.csv
|   |   |   ├── bbbp
|   |   |   ├── clintox
|   |   |   ├── esol
|   |   |   ├── freesolv
|   |   |   ├── hiv
|   |   |   ├── lipo
|   |   |   ├── qm9
|   |   |   ├── sider
|   |   |   ├── tox21
|   |   └── molformer splits

```


## Pretraining


## Finetuning


## Citations
