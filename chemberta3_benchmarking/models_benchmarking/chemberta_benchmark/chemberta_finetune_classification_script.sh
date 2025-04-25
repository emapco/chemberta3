python3 chemberta_finetune_classification.py\
        --datasets "bace,bbbp,tox21,hiv,sider,clintox" \
        --batch_size 32  \
        --epochs 100 \
        --pretrained_model_path '../../data/pretrained_model_checkpoints/pretrained_chemberta/chemberta-100M-mlm-4epochs' \
        --splits_name 'molformer_splits' \
        --learning_rate 3e-5 \