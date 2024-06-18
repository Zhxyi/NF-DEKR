<h1 align="left">Precise Localization for Anatomo-Physiological Hallmarks of the Cervical Spine by Using Neural Memory Ordinary Differential Equation</h1> 

## Project Overview
This project aims to achieve precise localization of anatomical and physiological hallmarks of the cervical spine using a Neural Memory Ordinary Differential Equation (nmODE). 

## Installation 
Download and install Miniconda from the https://docs.anaconda.com/free/miniconda/

Create and activate a new Conda environment.
```bash
conda create -n nm_ode_env python=3.8
conda activate nm_ode_env
```

We use PyTorch 2.0.1, and mmcv 2.0.0 for the experiments.
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv=2.0.0"
```

Install torchdiffeq, it is a library in PyTorch used for solving ordinary differential equations (ODEs) and partial differential equations (PDEs).
```bash
pip install torchdiffeq
```

## Usage
Organize your dataset in the following structure.
```txt
├── data
     ├── train images
     ├── val images
     └── annotations: 
     		  ├── keypoints_train.json
     		  └── keypoints_val.json
```

Training and testing.
```bash
python setyp.py install
# train
python tools\train.py <your config path>
```

```bash
# test
python tools\test.py <your config path>
```

Example.
```bash
python tools\train.py .\configs\body_2d_keypoint\nf_dekr\coco\NF-DEKR_hrnetw32.py
```

## Acknowledge
We acknowledge the excellent implementation from [mmpose](https://github.com/open-mmlab/mmpose).
