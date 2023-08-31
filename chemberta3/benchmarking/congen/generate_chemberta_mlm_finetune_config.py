import yaml
import os

regression_datasets = ['delaney', 'bace_r', 'clearance', 'lipo']
classification_datasets = ['bace_c', 'hiv', 'tox21', 'bbbp']
datasets = [*regression_datasets, *classification_datasets]
pretrain_models = {
    'zinc250k': {
        'pretrain_model_dir': 'runs/chemberta_mlm_pretrain_zinc250k'
    },
    'zinc1m': {
        'pretrain_model_dir': 'runs/chemberta_mlm_zinc1m'
    }
}


dataset_params = {
    'bace_c': {
        'n_classes':
            2,
        'data_dir':
            'data/bace_c-featurized/DummyFeaturizer/ScaffoldSplitter/BalancingTransformer'
    },
    'bace_r': {
        'data_dir':
            'data/bace_r-featurized/DummyFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True',
    },
    'bbbp': {
        'n_classes':
            2,
        'data_dir':
            'data/bbbp-featurized/DummyFeaturizer/ScaffoldSplitter/BalancingTransformer'
    },
    'clearance': {
        'data_dir':
            'data/clearance-featurized/DummyFeaturizer/ScaffoldSplitter/LogTransformer_transform_y_True'
    },
    'delaney': {
        'data_dir':
            'data/delaney-featurized/DummyFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True'
    },
    'hiv': {
        'n_classes':
            2,
        'data_dir':
            'data/hiv-featurized/DummyFeaturizer/ScaffoldSplitter/BalancingTransformer',
        'batch_size': 32,
    },
    'lipo': {
        'data_dir':
            'data/lipo-featurized/DummyFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True'
    },
    'tox21': {
        'data_dir':
            'data/tox21-featurized/DummyFeaturizer/ScaffoldSplitter/BalancingTransformer',
        'n_classes':
            2,
        'batch_size': 32,
    }
}

model_parameters = {
    'batch_size': 64,
    'learning_rate': 0.0001,
    'log_frequency': 10,
    'n_tasks': 1
}

other_parameters = {
    'checkpoint_interval': 4,
    'early_stopper': True,
    'train': True,
    'model_name': 'chemberta',
    'nb_epoch': 50,
    'patience': 10
}


def generate_config():
    commands = []
    for dataset in datasets:
        args = {}
        args = {**other_parameters}
        args_mp = model_parameters.copy()
        if dataset in classification_datasets:
            args_mp['mode'] = 'classification'
            args_mp['task'] = 'classification'
            args_mp['n_classes'] = dataset_params[dataset]['n_classes']
        else:
            args_mp['mode'] = 'regression'
            args_mp['task'] = 'regression'
        if 'batch_size' in dataset_params[dataset].keys():
            args_mp['batch_size'] = dataset_params[dataset]['batch_size']
        args['model_parameters'] = args_mp
        for value in ['train', 'test', 'valid']:
            args[f'{value}_data_dir'] = os.path.join(dataset_params[dataset]['data_dir'], f'{value}_dir') 
        for key, value in pretrain_models.items():
            args['pretrain_model_dir'] = value['pretrain_model_dir']
            filepath = f'chemberta_{key}_pretrain_{dataset}.yml'
            with open(os.path.join('../configs', filepath), 'w') as f:
                yaml.dump(args, f, default_flow_style=False)
            
            command = 'python3 benchmark.py --config configs/' + filepath 
            commands.append(command)

    commands = '\n'.join(commands)
    with open('chemberta_finetune.sh', 'w') as f:
        f.write(commands)

generate_config()
