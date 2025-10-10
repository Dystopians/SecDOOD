<h1 align="left">Secure On-Device Video OOD Detection Without Backpropagation</h1>

---
<div style="margin-top: 8px; font-family: Consolas, monospace; font-size: 16px;" align="center">

*SecDOOD is a secure cloud-device collaboration framework for efficient on-device OOD detection without requiring device-side backpropagation.*
<p align="center">
  <a href="https://arxiv.org/abs/2503.06166" target="_blank">
    <img src="imgs/logo_.png" alt="SecDOOD Logo" width="180">
  </a><br>
  <sub><code>Click the logo to read our paper on arXiv</code></sub>
</p>

</div>

<!-- <p align="center">
  <img src="imgs/logo_.png" alt="SecDOOD Logo" width="180">
</p> -->
<p align="center">
  <!-- ⭐ Stars badge -->
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/Dystopians/SecDOOD"/>
  <!-- 📄 License badge -->
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue"/>
  <!-- 🔖 ArXiv -->
  <a href="https://arxiv.org/abs/2503.06166"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2503.06166-B31B1B"/></a>
</p>


---

## SecDOOD
**Goal.** On-device video OOD detection with no device-side backpropagation, while preserving user privacy.

**Approach.** A cloud-hosted HyperNetwork generates device-specific classifier weights from an encrypted summary of device features. The device runs forward-only inference; no gradients or raw data leave the device.

**Workflow.**

1. **Cloud (offline):** Train base model $M_g$ and HyperNetwork $H$ on ID data.
    
2. **Device (online):** Extract features, score channel importance, homomorphically encrypt top $\alpha\%$ channels, mask the rest, and send encrypted subset.
    
3. **Cloud → Device:** Apply $H$ on encrypted features to produce $\Theta_d$; return to device for inference.
    

**Privacy & Efficiency.** Raw video and full features stay on device; the cloud only sees selectively encrypted channels. Dynamic channel sampling + selective encryption sharply reduce crypto and bandwidth costs with minimal accuracy loss.

**Compatibility.** Plug-and-play with common OOD scores (MSP, Energy, VIM) and modalities (RGB/flow/audio). Supports near- and far-OOD.

**Results.** On HMDB51, UCF101, EPIC-Kitchens, HAC, and Kinetics-600, SecDOOD increases AUROC and lowers FPR@95 versus local post-processing baselines—without any on-device training.

<p align="center">
  <img src="imgs/FigA.png" alt="SecDOOD Framework Overview" width="100%">
</p>
<p align="center"><em></em></p>

### Prepare Datasets
SecDOOD is based on five public action recognition datasets (HMDB51, UCF101, EPIC-Kitchens, HAC, and Kinetics-600).
1. Download HMDB51 video data from [link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) and extract. Download HMDB51 optical flow data from [link](https://huggingface.co/datasets/hdong51/MultiOOD/blob/main/hmdb51_flow_mp4.tar.gz) and extract. The directory structure should be modified to match:
   <details>
   <summary>Click for details...</summary>

   ```
   HMDB51
   ├── video
   |   ├── catch
   |   |   ├── *.avi
   |   ├── climb
   |   |   ├── *.avi
   |   |── ...


   ├── flow
   |   ├── *_flow_x.mp4
   |   ├── *_flow_y.mp4
   |   ├── ...
   ```

   </details>

2. Download UCF101 video data from [link](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) and extract. Download UCF101 optical flow data from [link](https://huggingface.co/datasets/hdong51/MultiOOD/blob/main/ucf101_flow_mp4.tar.gz) and extract. The directory structure should be modified to match:
   <details>
   <summary>Click for details...</summary>

   ```
   UCF101
   ├── video
   |   ├── *.avi
   |   |── ...


   ├── flow
   |   ├── *_flow_x.mp4
   |   ├── *_flow_y.mp4
   |   ├── ...
   ```

   </details>

3. Download EPIC-Kitchens video and optical flow data by

   ```
   bash utils/download_epic_script.sh
   ```

   Download audio data from [link](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/blob/main/EPIC-KITCHENS-audio.zip).

   Unzip all files and the directory structure should be modified to match:
   <details>
   <summary>Click for details...</summary>

   ```
   EPIC-KITCHENS
   ├── rgb
   |   ├── train
   |   |   ├── D3
   |   |   |   ├── P22_05.wav
   |   |   |   ├── P22_05
   |   |   |   |     ├── frame_0000000000.jpg
   |   |   |   |     ├── ...
   |   |   |   ├── P22_06
   |   |   |   ├── ...
   |   ├── test
   |   |   ├── D3
   |   |   |   ├── P22_01.wav
   |   |   |   ├── P22_01
   |   |   |   |     ├── frame_0000000000.jpg
   |   |   |   |     ├── ...
   |   |   |   ├── P22_02
   |   |   |   ├── ...

   ├── flow
   |   ├── train
   |   |   ├── D3
   |   |   |   ├── P22_05
   |   |   |   |     ├── frame_0000000000.jpg
   |   |   |   |     ├── ...
   |   |   |   ├── P22_06
   |   |   |   ├── ...
   |   ├── test
   |   |   ├── D3
   |   |   |   ├── P22_01
   |   |   |   |     ├── frame_0000000000.jpg
   |   |   |   |     ├── ...
   |   |   |   ├── P22_02
   |   |   |   ├── ...
   ```

   </details>

4. Download HAC video, audio and optical flow data from [link](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/tree/main) and extract. The directory structure should be modified to match:
   <details>
   <summary>Click for details...</summary>

   ```
   HAC
   ├── human
   |   ├── videos
   |   |   ├── ...
   |   ├── flow
   |   |   ├── ...
   |   ├── audio
   |   |   ├── ...

   ├── animal
   |   ├── videos
   |   |   ├── ...
   |   ├── flow
   |   |   ├── ...
   |   ├── audio
   |   |   ├── ...

   ├── cartoon
   |   ├── videos
   |   |   ├── ...
   |   ├── flow
   |   |   ├── ...
   |   ├── audio
   |   |   ├── ...
   ```

   </details>

5. Download Kinetics-600 video data by

   ```
   wget -i utils/filtered_k600_train_path.txt
   ```

   Extract all files and get audio data from video data by

   ```
   python utils/generate_audio_files.py
   ```

   Download Kinetics-600 optical flow data (kinetics600_flow_mp4_part_*) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main) and extract (run `cat kinetics600_flow_mp4_part_* > kinetics600_flow_mp4.tar.gz` and then `tar -zxvf kinetics600_flow_mp4.tar.gz`).

   Unzip all files and the directory structure should be modified to match:
   <details>
   <summary>Click for details...</summary>

   ```
   Kinetics-600
   ├── video
   |   ├── acting in play
   |   |   ├── *.mp4
   |   |   ├── *.wav
   |   |── ...


   ├── flow
   |   ├── acting in play
   |   |   ├── *_flow_x.mp4
   |   |   ├── *_flow_y.mp4
   |   ├── ...
   ```

   </details>

### Dataset Splits
The splits for Multimodal Near-OOD and Far-OOD Benchmarks are provided under `HMDB-rgb-flow/splits/` for HMDB51, UCF101, HAC, and Kinetics-600, and under `EPIC-rgb-flow/splits/` for EPIC-Kitchens.

---

## Methodology

<p align="center">
  <img src="imgs/FigB_round.png" alt="SecDOOD Methodology" width="80%">
</p>

<p><strong>SecDOOD</strong> is a cloud–device collaborative pipeline for on-device video OOD detection with <em>zero device-side backpropagation</em>. A cloud-hosted <em>HyperNetwork</em> <code>H</code> is trained on in-distribution (ID) data and returns <em>device-specific</em> classifier parameters <code>Θ<sub>d</sub></code>; the device performs forward-only inference.</p>

<ol>
  <li><strong>Device feature extraction.</strong> Compute intermediate features and rank channels via Shapley-style contribution estimates.</li>
  <li><strong>Selective encryption &amp; upload.</strong> Encrypt the top <strong>50%</strong> most informative channels (mask the rest) and send only this encrypted subset to the cloud.</li>
  <li><strong>Cloud personalization.</strong> Evaluate <code>H</code> on the encrypted features to produce <code>Θ<sub>d</sub></code>. Raw videos and full feature tensors never leave the device.</li>
  <li><strong>On-device inference.</strong> Decrypt and inject <code>Θ<sub>d</sub></code> into the local head and compute OOD scores (e.g., MSP, Energy, VIM); inference is forward-only.</li>
</ol>

<p><em>Privacy &amp; efficiency.</em> Dynamic channel selection plus selective encryption reduces bandwidth and cryptographic cost while preserving accuracy, enabling plug-and-play deployment on resource-constrained devices.</p>

---

## Code
The code was tested using `Python 3.10.4`, `torch 1.11.0+cu113` and `NVIDIA GeForce RTX 3090`. More dependencies are in `requirement.txt`.

### Prepare

#### Download Pretrained Weights
1. Download SlowFast model for RGB modality [link](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) and place under the `HMDB-rgb-flow/pretrained_models` and `EPIC-rgb-flow/pretrained_models` directory
   
2. Download SlowOnly model for Flow modality [link](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) and place under the `HMDB-rgb-flow/pretrained_models` and `EPIC-rgb-flow/pretrained_models` directory

3. Download Audio model [link](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), rename it as `vggsound_avgpool.pth.tar` and place under the `HMDB-rgb-flow/pretrained_models` and `EPIC-rgb-flow/pretrained_models`  directory
   
### Multimodal Near-OOD Benchmark
#### HMDB51 25/26

<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Near-OOD model for HMDB:

```
python Train.py --near_ood --dataset 'HMDB' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/HMDB51/'
```

You can also download our provided checkpoints (`HMDB_near_ood_baseline.pt`, `HMDB_near_ood_a2d.pt`, and `HMDB_near_ood_a2d_npmix.pt`) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main/checkpoints).

Save the evaluation files for HMDB (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python Test.py --bsz 16 --num_workers 2 --near_ood --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_near_ood_a2d_npmix.pt'
```

Evaluation for HMDB (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --path 'HMDB-rgb-flow/'
```

</details>

#### UCF101 50/51
<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Near-OOD model for UCF:

```
python Train.py --near_ood --dataset 'UCF' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/UCF101/'
```

You can also download our provided checkpoints (`UCF_near_ood_baseline.pt`, `UCF_near_ood_a2d.pt`, and `UCF_near_ood_a2d_npmix.pt`) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main/checkpoints).

Save the evaluation files for UCF (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python Test.py --bsz 16 --num_workers 2 --near_ood --dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/path/to/UCF_near_ood_a2d_npmix.pt'
```

Evaluation for UCF (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'UCF' --path 'HMDB-rgb-flow/'
```

</details>

#### EPIC-Kitchens 4/4
<details>
<summary>Click for details...</summary>

```
cd EPIC-rgb-flow/
```

Train the Near-OOD baseline model for EPIC:

```
python Train_Epic.py --dataset 'EPIC' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/EPIC-Kitchens/'
```

Train the Near-OOD model using A2D for EPIC:

```
python Train_Epic.py --dataset 'EPIC' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 1.0 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/EPIC-Kitchens/'
```

Train the Near-OOD model using A2D and NP-Mix for EPIC:

```
python Train_Epic.py --dataset 'EPIC' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/EPIC-Kitchens/'
```

You can also download our provided checkpoints (`EPIC_near_ood_baseline.pt`, `EPIC_near_ood_a2d.pt`, and `EPIC_near_ood_a2d_npmix.pt`) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main/checkpoints).

Save the evaluation files for EPIC (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python Test_near.py --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/EPIC_near_ood_a2d_npmix.pt'
```

Evaluation for EPIC (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'EPIC' --path 'EPIC-rgb-flow/'
```

</details>

#### Kinetics-600 129/100
<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Near-OOD baseline model for Kinetics:

```
python Train.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics-600/'
```

Train the Near-OOD model using A2D for Kinetics:

```
python Train.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 1.0 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics-600/'
```

Train the Near-OOD model using A2D and NP-Mix for Kinetics:

```
python Train.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics-600/'
```

You can also download our provided checkpoints (`Kinetics_near_ood_baseline.pt`, `Kinetics_near_ood_a2d.pt`, and `Kinetics_near_ood_a2d_npmix.pt`) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main/checkpoints).

Save the evaluation files for Kinetics (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python Test.py --bsz 16 --num_workers 2 --near_ood --dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_near_ood_a2d_npmix.pt'
```

Evaluation for Kinetics (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'Kinetics' --path 'HMDB-rgb-flow/'
```

</details>

### Multimodal Far-OOD Benchmark
#### HMDB51 as ID

<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Far-OOD baseline model for HMDB:

```
python Train.py --dataset 'HMDB' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/HMDB51/'
```

Train the Far-OOD model using A2D and NP-Mix for HMDB:

```
python Train.py --dataset 'HMDB' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 50 --appen '' --save_best --save_checkpoint --datapath '/path/to/HMDB51/'
```

You can also download our provided checkpoints (`HMDB_far_ood_baseline.pt` and `HMDB_far_ood_a2d_npmix.pt`) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main/checkpoints).

Save the evaluation files for HMDB (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`, same for other datasets):
```
python Test.py --bsz 16 --num_workers 2 --dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'
```

Save the evaluation files for UCF:
```
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'
```

Save the evaluation files for HAC:
```
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'HAC' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'
```

Save the evaluation files for Kinetics:
```
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'
```

Save the evaluation files for EPIC:
```
cd EPIC-rgb-flow/
```
```
python Test_far.py --bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/HMDB_far_ood_a2d_npmix.pt'
```

Evaluation for UCF (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint, change `--ood_dataset` to UCF, EPIC, HAC, or Kinetics):
```
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'HMDB' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/'
```

</details>

#### Kinetics as ID

<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Far-OOD baseline model for Kinetics:

```
python Train.py --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --nepochs 10 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics-600/'
```

Train the Far-OOD model using A2D and NP-Mix for Kinetics:

```
python Train.py --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 3 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.1 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.1 --ood_entropy_ratio 0.1 --nepochs 10 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics-600/'
```

You can also download our provided checkpoints (`Kinetics_far_ood_baseline.pt` and `Kinetics_far_ood_a2d_npmix.pt`) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main/checkpoints).

Save the evaluation files for Kinetics (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`, same for other datasets):
```
python Test.py --bsz 16 --num_workers 2 --dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_far_ood_a2d_npmix.pt'
```

Save the evaluation files for HMDB:
```
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'HMDB' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_far_ood_a2d_npmix.pt'
```

Save the evaluation files for UCF:
```
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'UCF' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_far_ood_a2d_npmix.pt'
```

Save the evaluation files for HAC:
```
python Test.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'HAC' --appen 'a2d_npmix_best_' --resumef '/path/to/cKinetics_far_ood_a2d_npmix.pt'
```

Save the evaluation files for EPIC:
```
cd EPIC-rgb-flow/
```
```
python Test_far.py --bsz 16 --num_workers 2 --far_ood --dataset 'Kinetics' --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_far_ood_a2d_npmix.pt'
```

Evaluation for UCF (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint, change `--ood_dataset` to UCF, EPIC, HAC, or HMDB):
```
python eval_video_flow_far_ood.py --postprocessor msp --appen 'a2d_npmix_best_' --dataset 'Kinetics' --ood_dataset 'UCF' --path 'HMDB-rgb-flow/'
```

</details>

### Multimodal Near-OOD Benchmark with Video, Audio, and Optical Flow

#### EPIC-Kitchens 4/4
<details>
<summary>Click for details...</summary>

```
cd EPIC-rgb-flow/
```

Train the Near-OOD baseline model for EPIC:

```
python Train_audio_epic.py --dataset 'EPIC' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/EPIC-Kitchens/'
```

Train the Near-OOD model using A2D and NP-Mix for EPIC:

```
python Train_audio_epic.py --dataset 'EPIC' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/EPIC-Kitchens/'
```

You can also download our provided checkpoints (`EPIC_near_ood_vfa_baseline.pt` and `EPIC_near_ood_vfa_a2d_npmix.pt`) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main/checkpoints).

Save the evaluation files for EPIC (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python Test_audio_epic.py --bsz 16 --num_workers 2  --ood_dataset 'EPIC' --appen 'a2d_npmix_best_' --resumef '/path/to/EPIC_near_ood_vfa_a2d_npmix.pt'
```

Evaluation for EPIC (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_a2d_npmix_best_' --dataset 'EPIC' --path 'EPIC-rgb-flow/'
```

</details>

#### Kinetics-600 129/100
<details>
<summary>Click for details...</summary>

```
cd HMDB-rgb-flow/
```

Train the Near-OOD baseline model for Kinetics:

```
python Train_audio.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics-600/'
```

Train the Near-OOD model using A2D and NP-Mix for Kinetics:

```
python Train_audio.py --near_ood --dataset 'Kinetics' --lr 0.0001 --seed 0 --bsz 16 --num_workers 10 --start_epoch 10 --use_single_pred --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 20 --appen '' --save_best --save_checkpoint --datapath '/path/to/Kinetics-600/'
```

You can also download our provided checkpoints (`Kinetics_near_ood_vfa_baseline.pt` and `Kinetics_near_ood_vfa_a2d_npmix.pt`) from [link](https://huggingface.co/datasets/hdong51/MultiOOD/tree/main/checkpoints).

Save the evaluation files for Kinetics (to save evaluation files for ASH or ReAct, you should also run following line with options `--use_ash` or `--use_react`):
```
python Test_audio.py --bsz 16 --num_workers 2 --near_ood --dataset 'Kinetics' --appen 'a2d_npmix_best_' --resumef '/path/to/Kinetics_near_ood_a2d_npmix.pt'
```

Evaluation for Kinetics (change `--postprocessor` to different score functions, for VIM you should also pass `--resume_file checkpoint.pt`, where checkpoint.pt is the trained checkpoint):
```
python eval_video_flow_near_ood.py --postprocessor msp --appen 'vfa_a2d_npmix_best_' --dataset 'Kinetics' --path 'HMDB-rgb-flow/'
```

</details>

---

## Contact
If you have any questions, please send an email to peilinca@usc.edu or make an issue :)

---

## Citation
```
@article{li2025secure,
  title={Secure on-device video ood detection without backpropagation},
  author={Li, Shawn and Cai, Peilin and Zhou, Yuxiao and Ni, Zhiyu and Liang, Renjie and Qin, You and Nian, Yi and Tu, Zhengzhong and Hu, Xiyang and Zhao, Yue},
  journal={arXiv preprint arXiv:2503.06166},
  year={2025}
}
```