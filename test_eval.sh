#!/bin/bash
# run_eval.sh

cd HMDB-rgb-flow

# near_ood
python Test.py --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_near_ood_a2d_npmix.pt'
python Test.py --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/path/to/UCF_near_ood_a2d_npmix.pt'
python Test.py --bsz 16 --num_workers 2 --near_ood --dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_near_ood_a2d_npmix.pt'

# HMDB far_ood
python Test.py --bsz 16 --num_workers 2 --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'HAC' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'

# Kinetics far_ood
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'

python Test.py --bsz 16 --num_workers 2 --dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_far_ood_a2d_npmix.pt'
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_far_ood_a2d_npmix.pt'
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_far_ood_a2d_npmix.pt'
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'HAC' --appen 'a2d_npmix_best_' --resumef '/path/to/cKinetics_far_ood_a2d_npmix.pt'

cd ..

# EPIC
cd EPIC-rgb-flow/

python Test_far.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_far_ood_a2d_npmix.pt' --use_ash
python Test_far.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt' --use_react
python Test_near.py --bsz 16 --num_workers 2 --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/EPIC_near_ood_a2d_npmix.pt'

cd ..

python eval_video_flow_near_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'HMDB' --path 'HMDB-rgb-flow/'
python eval_video_flow_near_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'UCF' --path 'HMDB-rgb-flow/'
python eval_video_flow_near_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'EPIC' --path 'EPIC-rgb-flow/'
python eval_video_flow_near_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'Kinetics' --path 'HMDB-rgb-flow/'

python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/'
python eval_video_flow_far_ood.py --postprocessor react --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'EPIC' --path 'EPIC-rgb-flow/'
python eval_video_flow_far_ood.py --postprocessor Mahalanobis --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'HAC' --path 'EPIC-rgb-flow/'

python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'Kinetics' --ood_dataset 'HMDB' --path 'HMDB-rgb-flow/'
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'Kinetics' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/'
python eval_video_flow_far_ood.py --postprocessor gen --appen 'a2d_npmix_best_' --dataset 'Kinetics' --ood_dataset 'HAC' --path 'HMDB-rgb-flow/'

# EPIC-rgb-flow/saved_files/ to HMDB-rgb-flow/saved_files/
mv EPIC-rgb-flow/saved_files/* HMDB-rgb-flow/saved_files/

python eval_video_flow_far_ood.py --postprocessor ash --appen 'a2d_npmix_best_' --dataset 'Kinetics' --ood_dataset 'EPIC' --path 'HMDB-rgb-flow/' --use_ash
python eval_video_flow_far_ood.py --postprocessor react --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'Kinetics' --path 'HMDB-rgb-flow/' --use_react

cd ..
