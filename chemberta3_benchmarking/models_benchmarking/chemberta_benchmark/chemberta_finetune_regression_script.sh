python3 chemberta_finetune_regression.py\
        --datasets "esol" \
        --transform True \
        --batch_size 32  \
        --epochs 100 \
        --pretrained_model_path '' \
        --splits_name 'deepchem_splits' \
        --learning_rate 3e-5 \

# please provide the pretrained model path