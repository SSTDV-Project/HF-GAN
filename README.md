HF-GAN
===
[[Preprint-v1](https://arxiv.org/abs/2406.14954v1)]
[[Preprint-v2](https://arxiv.org/abs/2406.14954)]
[[Paper](https://doi.org/10.1016/j.compmedimag.2026.102838)]

This is the official implementation of "Structured feature disentanglement for high-fidelity unified brain MRI synthesis".

## How to run
```bash
### Clone the repository
$ git clone https://github.com/SSTDV-Project/HF-GAN.git
$ cd HF-GAN

### Install & activate environment
$ conda env create -f environment.yml
$ conda activate syn

### Set up the configuration of accelerate
$ accelerate config
```
We suggest the following folder structure for training
```
data/
--- BraTS/
------ train/
------ valid/
--- IXI/
------ train/
------ valid/
```
We use the hdf5 format for training.
```
xxx.hdf5 = {image: 2D slices of C modalities (CxHxW)}
```

## Dataset

BraTS 2018: https://www.med.upenn.edu/sbia/brats2018/data.html

IXI: https://brain-development.org/ixi-dataset

## Training
```
For the BraTS dataset
  CUDA_VISIBLE_DEVICES=0 python3 train_BraTS.py --dataset data/BraTS --identifier "name of the checkpoints"
For the IXI dataset
  CUDA_VISIBLE_DEVICES=0 python3 train_IXI.py --dataset data/IXI --identifier "name of the checkpoints"
```

## Tested environment
* OS: Ubuntu 20.04
* GPU: NVIDIA GeForce RTX 3090
* GPU Driver: 535.129.03
* Host CUDA version: 12.2

## Citation

```bibtex
@article{cho2026structured,
title = {Structured feature disentanglement for high-fidelity unified brain MRI synthesis},
journal = {Computerized Medical Imaging and Graphics},
pages = {102838},
year = {2026},
issn = {0895-6111},
doi = {https://doi.org/10.1016/j.compmedimag.2026.102838},
url = {https://www.sciencedirect.com/science/article/pii/S0895611126001412},
author = {Jihoon Cho and Jonghye Woo and Jinah Park}
}
```


## Acknowledgement

>This work was supported by Institute for Information \& communications Technology Promotion(IITP) grant funded by the Korea government (MSIT) (No.00223446, Development of object-oriented synthetic data generation and evaluation methods).
