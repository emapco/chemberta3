../../../../.venv/bin/python prepare_data.py \
    --split_type 'deepchem' \
    --datasets 'bbbp,bace_classification,clintox,hiv,tox21,sider,delaney,freesolv,lipo,clearance,bace_regression,antimalarial,cocrystal,covid19,adme_microsom_stab_h,adme_microsom_stab_r,adme_permeability,adme_ppb_h,adme_ppb_r,adme_solubility,astrazeneca_cl,astrazeneca_logd74,astrazeneca_ppb,astrazeneca_solubility' \
    --featurizers 'dummy' \
    --data_dir ./../datasets/deepchem_splits \
    --feat_dir ./../featurized_datasets/deepchem_splits \
    --max_smiles_len 200
