python3 infomax3d_classification_benchmark.py\
        --datasets "bace,bbbp,tox21" \
        --batch_size 64 \
        --hidden_dim 64 \
        --target_dim 10 \
        --epochs 100 \
        --pretrained_model_path '../../data/pretrained_model_checkpoints/pretrained_infomax3d/pretrain_infomax3D_250K/checkpoint1.pt' \
        --splits_name 'deepchem_splits' \
        --learning_rate 3e-05 \
