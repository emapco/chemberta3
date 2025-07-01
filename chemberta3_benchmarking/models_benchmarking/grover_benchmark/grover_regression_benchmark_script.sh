python3 grover_regression_benchmark.py\
        --datasets "freesolv,bace_regression,clearance,esol" \
        --splits_name 'deepchem_splits' \
        --batch_size 100 \
        --vocab_data_path '../../data/dummy_featurized/zinc250k_dummy_featurized' \
        --node_fdim 151 \
        --edge_fdim 165 \
        --feature_dim 2048 \
        --hidden_size 128 \
        --functional_group_size 85 \
        --learning_rate 3e-5 \
        --epochs 100 \
        --pretrained_model_path '' \
        --transform

# please provide the pretrained_model_path