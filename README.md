HF-GAN
===
[[Preprint]()]

This is the official implementation of "A Unified Framework for Synthesizing Multisequence Brain MRI via Hybrid Fusion".

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
### Training
```
For the BraTS dataset
  CUDA_VISIBLE_DEVICES=0 python3 train_BraTS.py --dataset data/BraTS --identifier "name of the checkpoints"
For the IXI dataset
  CUDA_VISIBLE_DEVICES=0 python3 train_BraTS.py --dataset data/IXI --identifier "name of the checkpoints"
```

## Tested environment
* OS: Ubuntu 20.04
* GPU: NVIDIA GeForce RTX 3090
* GPU Driver: 535.129.03
* Host CUDA version: 12.2

## Acknowledgement

> This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.00223446, Development of object-oriented synthetic data generation and evaluation methods)
