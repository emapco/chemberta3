python3 rf_singletask_classification_benchmark.py\
        --datasets "bace" \
        --splits_name 'deepchem_splits' \
        --bootstrap False \
        --criterion gini \
        --min_samples_split 32 \
        --n_estimators 100 \