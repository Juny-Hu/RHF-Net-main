# RHF-Net

RHF-Net is a paired RGB–hyperspectral image classification network for four picking-period classes of Lu'an Guapian tea. The repository contains the clean model definition, a desktop inference GUI, five-fold weights for RGB-only, PCA20-HSI-only, and paired RGB–HSI prediction, the PCA20 projection parameters, and eight example pairs.

## Repository layout

```text
RHF-Net-main/
├── RHF-Net.py
├── gui.py
└── example/
    ├── rgb/test1-p1.npy ... test8-p4.npy
    └── hyper/test1-p1.npy ... test8-p4.npy
```

## Environment

Python 3.11 or newer is recommended. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run the GUI

From the repository root, run:

```bash
python gui.py
```

The GUI supports RGB-only, hyperspectral-only, and paired RGB–hyperspectral prediction. Select matching files from `example/rgb` and `example/hyper` to test the paired mode. The hyperspectral input is expected to contain 20 PCA channels. The five fold probabilities are averaged before the final class is reported.

The model weights and PCA transformation file are distributed separately. After downloading them, place the files in the following structure, or provide alternative paths with the corresponding command-line arguments:

```text
weights/rgb/fold_1..fold_5/best.pt
weights/hsi/fold_1..fold_5/best.pt
weights/fusion/fold_1..fold_5/best.pt
artifacts/pca20_training_transform.npz
```
## Model weights and PCA transformation file
```text
https://pan.baidu.com/s/1KKstmW4mgEDON9-_MMUjyQ](https://pan.baidu.com/s/1KKstmW4mgEDON9-_MMUjyQ
key：i236
```
The repository intentionally does not include these large binary files.

## Input format

RGB files are NumPy arrays with three channels in CHW or HWC layout. Hyperspectral files are NumPy arrays with 20 PCA channels in CHW or HWC layout. The GUI resizes inputs to 224 × 224 pixels and applies the same normalization used during model training.
