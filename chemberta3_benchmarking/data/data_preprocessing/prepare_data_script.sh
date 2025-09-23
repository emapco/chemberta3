../../../../.venv/bin/python prepare_data.py \
    --split_type 'deepchem' \
    --datasets 'bbbp,bace_classification,clintox,hiv,tox21,sider,delaney,freesolv,lipo,clearance,bace_regression' \
    --featurizers 'dummy' \
    --data_dir ./../datasets/deepchem_splits \
    --feat_dir ./../featurized_datasets/deepchem_splits \
    --max_smiles_len 200
