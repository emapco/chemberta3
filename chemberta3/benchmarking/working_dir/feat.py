import argparse
import logging
import deepchem as dc
import ray
from data_utils import RayDataset

logging.basicConfig(filename='ray.log', level=logging.INFO)

FEATURIZER_MAPPING = {
    "molgraphconv":
    dc.feat.MolGraphConvFeaturizer(use_edges=True),
    "dummy":
    dc.feat.DummyFeaturizer(),
    "grover":
    dc.feat.GroverFeaturizer(features_generator=dc.feat.CircularFingerprint()),
    "rdkit-conformer":
    dc.feat.RDKitConformerFeaturizer(),
    "snap":
    dc.feat.SNAPFeaturizer(),
}


def get_paths_from_args(args):
    result_path = 's3://chemberta3/featurized_data/' + args.dataset + '/' + args.featurizer
    large_datasets = ['zinc100m', 'zinc250m', 'zinc500m', 'zinc1b']
    if args.dataset in large_datasets:
        if args.dataset == 'zinc100m':
            upper_limit = 10
        if args.dataset == 'zinc250m':
            upper_limit = 25
        if args.dataset == 'zinc500m':
            upper_limit = 50
        if args.dataset == 'zinc1b':
            upper_limit = 100
        chunks = []
        for i in range(0, upper_limit):
            chunks.append('chunk_00' + str(i) + '.csv')
        csv_paths = []
        for chunk_path in chunks:
            csv_paths.append('s3://chemberta3/datasets/zinc20/csv/' +
                             chunk_path)
        return csv_paths, result_path
    else:
        csv_path = 's3://chemberta3/datasets/' + args.dataset + '.csv'
        return csv_path, result_path


def test_is_empty_path(path):
    import boto3
    bucket_name = path.strip('s3://').split('/')[0]
    folder_name = path.strip('s3://' + bucket_name + '/')

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    count = bucket.objects.filter(Prefix=folder_name)
    assert len(list(count)) == 0, f'Result path is not empty {path}'


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='zinc250k')
    argparser.add_argument('--featurizer', type=str, default='dummy')
    argparser.add_argument('--column', type=str, default='smiles')
    argparser.add_argument('--label-column', type=str, default=None)
    args = argparser.parse_args()

    if args.dataset not in [
            'zinc5k', 'zinc250k', 'zinc1m', 'zinc10m', 'zinc100m', 'zinc250m',
            'zinc500m', 'zinc1b'
    ]:
        raise ValueError(
            'Featurization is currently supported only for zinc datasets')
    csv_path, result_path = get_paths_from_args(args)
    test_is_empty_path(result_path)

    featurizer = FEATURIZER_MAPPING[args.featurizer]
    ray.data.DataContext.get_current(
    ).execution_options.verbose_progress = True
    large_datasets = ['zinc100m', 'zinc250m', 'zinc500m', 'zinc1b']

    # Specifies the level of parallelism for the operation. Ray will process the CSV file in 100 parallel tasks,
    # which can improve loading speed, especially for large files.
    ds = ray.data.read_csv(csv_path, parallelism=100)

    # Redistributes the dataset into 10,000 partitions in case of large datasets (>100M). This is typically
    # done to balance the data more evenly across computational resources or to adjust the partition size
    # for subsequent processing.
    if args.dataset in large_datasets:
        ds = ds.repartition(10_000)
    ds = RayDataset(ds)

    ds.featurize(featurizer=featurizer, column='smiles')
    ds.write(result_path, columns=['x', 'smiles'])
    print('wrote data')
