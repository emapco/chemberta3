python3 dmpnn_regression_benchmark.py\
        --datasets "lipo,freesolv,bace_regression,clearance,esol" \
        --batch_size 128  \
        --epochs 100 \
        --splits_name 'deepchem_splits' \
        --learning_rate 0.001 \
        --transform