python3 rf_regression_benchmark.py\
        --datasets "esol,freesolv,lipo,bace_regression,clearance" \
        --transform True \
        --splits_name 'deepchem_splits' \
        --bootstrap True \
        --criterion squared_error \
        --min_samples_split 2 \
        --n_estimators 100 \