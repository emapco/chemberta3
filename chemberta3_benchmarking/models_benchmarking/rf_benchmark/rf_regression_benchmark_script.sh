python3 rf_regression_benchmark.py\
        --datasets "esol,freesolv,lipo" \
        --splits_name 'molformer_splits' \
        --bootstrap True \
        --criterion squared_error \
        --min_samples_split 2 \
        --n_estimators 10 \