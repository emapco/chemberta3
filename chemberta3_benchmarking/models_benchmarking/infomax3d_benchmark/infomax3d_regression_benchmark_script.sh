python3 infomax3d_regression_benchmark.py\
        --datasets "lipo,freesolv,bace_regression,clearance,esol" \
        --batch_size 32 \
        --hidden_dim 64 \
        --target_dim 10 \
        --epochs 500 \
        --pretrained_model_path '' \
        --splits_name 'deepchem_splits' \
        --learning_rate 3e-05 \
        --transform

# Please provide the pretrained model path