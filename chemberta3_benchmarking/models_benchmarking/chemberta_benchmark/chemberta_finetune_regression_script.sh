../../../../.venv/bin/python chemberta_finetune_regression.py\
        --datasets "delaney,bace_regression,freesolv,lipo,clearance" \
        --batch_size 32  \
        --epochs 100 \
        --pretrained_model_path 'Derify/ChemBERTa_augmented_pubchem_13m' \
        --splits_name 'deepchem_splits' \
        --learning_rate 3e-5 \
        --transform
