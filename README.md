<h1 align="left">Precise Localization for Anatomo-Physiological Hallmarks of the Cervical Spine by Using Neural Memory Ordinary Differential Equation</h1> 

## Usage

We use PyTorch 2.0.1, and mmcv 2.0.0 for the experiments.
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
```

After install the two repos, install timm and einops, i.e.,
```bash
pip install timm==0.4.9 einops
```

After downloading the pretrained models, please conduct the experiments by running

```bash
