# chemberta3
ChemBERTa-3 Repo


# Benchmarking
The `benchmark.py` script provides the ability to benchmark models against different downstream datasets.

Example command:
```python benchmark.py --dataset_name=delaney --model_name=infograph --featurizer_name=molgraphconv --checkpoint=checkpoint5.pt```

### Benchmarking random forest

```
python benchmark.py --dataset_name=delaney --model_name=random_forest --featurizer_name=ecfp
python benchmark.py --dataset_name=bace_classification --model_name=random_forest --featurizer_name=ecfp
python benchmark.py --dataset_name=bace_regression --model_name=random_forest --featurizer_name=ecfp
```
