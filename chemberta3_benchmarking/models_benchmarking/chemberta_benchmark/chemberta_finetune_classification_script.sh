../../../../.venv/bin/python chemberta_finetune_classification.py\
        --datasets "bace_classification,bbbp,tox21,hiv,sider,clintox" \
        --batch_size 32  \
        --epochs 100 \
        --pretrained_model_path 'Derify/ChemBERTa_augmented_pubchem_13m' \
        --splits_name 'deepchem_splits' \
        --learning_rate 3e-5
