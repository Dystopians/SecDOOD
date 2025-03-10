#!/bin/bash
# train.sh
cd HMDB-rgb-flow/

# HMDB
python Train.py --near_ood --dataset 'HMDB' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 \
    --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger \
    --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '' --save_best --save_checkpoint \
    --datapath '/path/to/HMDB51/'

# UCF
python Train.py --near_ood --dataset 'UCF' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 \
    --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger \
    --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '' --save_best --save_checkpoint \
    --datapath '/path/to/UCF101/'

# Kinetics
python Train.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 \
    --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger \
    --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 20 --appen '' --save_best --save_checkpoint \
    --datapath '/path/to/Kinetics-600/'

# HMDB far
python Train.py --dataset 'HMDB' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 \
    --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger \
    --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 50 --appen '' --save_best --save_checkpoint \
    --datapath '/path/to/HMDB51/'

# Kinetics far
python Train.py --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 3 \
    --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger \
    --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 10 --appen '' --save_best --save_checkpoint \
    --datapath '/path/to/Kinetics-600/'

cd ..

# EPIC
cd EPIC-rgb-flow/
python Train_Epic.py --dataset 'EPIC' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 \
    --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger \
    --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 20 --appen '' --save_best --save_checkpoint \
    --datapath '/path/to/EPIC-Kitchens/'

cd ..
