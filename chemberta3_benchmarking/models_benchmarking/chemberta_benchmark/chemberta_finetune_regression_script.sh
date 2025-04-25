python3 chemberta_finetune_regression.py\
        --datasets "esol,freesolv,lipo" \
        --batch_size 128  \
        --epochs 100 \
        --pretrained_model_path '../../data/pretrained_model_checkpoints/pretrained_chemberta/chemberta-100M-mlm-4epochs' \
        --splits_name 'molformer_splits' \
        --learning_rate 3e-5 \
