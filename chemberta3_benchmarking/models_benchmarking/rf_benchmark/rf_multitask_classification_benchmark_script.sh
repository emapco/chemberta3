python3 rf_multitask_classification_benchmark.py\
        --datasets "sider,tox21,clintox" \
        --splits_name 'deepchem_splits' \
        --bootstrap False \
        --criterion gini \
        --min_samples_split 32 \
        --n_estimators 100 \