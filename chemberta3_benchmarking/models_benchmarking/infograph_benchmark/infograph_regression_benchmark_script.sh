python3 infograph_regression_benchmark.py\
        --datasets "clearance,bace_regression,esol,freesolv,lipo" \
        --transform True \
        --batch_size 128 \
        --num_feat 30 \
        --edge_dim 11 \
        --num_gc_layers 4 \
        --epochs 100 \
        --pretrained_model_path '' \
        --splits_name 'deepchem_splits' \
        --learning_rate 0.001 \

# please provide the pretrained model path