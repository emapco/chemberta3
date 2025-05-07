python3 infograph_classification_benchmark.py\
        --datasets "bace,bbbp,tox21,clintox,hiv,sider" \
        --batch_size 128 \
        --num_feat 30 \
        --edge_dim 11 \
        --num_gc_layers 4 \
        --epochs 100 \
        --pretrained_model_path '../../data/pretrained_model_checkpoints/pretrained_infograph/pretrain-infograph-250k-50epochs-gclayers4/checkpoint1.pt' \
        --splits_name 'deepchem_splits' \
        --learning_rate 0.001 \
