# Debiasing-based Few-Shot Learning (DFSL)

PyTorch implementation for the paper: Debiasing-based Few-Shot Learning

## Dependencies
* python 3.6.5
* numpy 1.16.0
* torch 1.8.0
* tqdm 4.57.0
* scipy 1.5.4
* torchvision 0.9.0

## Overview
The problem of feature bias presents a major challenge for Few-Shot Learning tasks. There are two sources of this bias: (1) during the pretraining process, cross-class bias can be induced for unseen classes, and (2) few labeled examples are not informative to represent their respective classes, causing instance-level feature bias. To alleviate the bias, we propose a debiasing autoencoder that is trained with biased and unbiased features extracted from samples belonging to the validation dataset, where the feature extractors are trained with and without these classes, respectively. The resulting model is further fine-tuned on the few support samples for adaptation to novel tasks. Furthermore, we contend that distribution-based representations are less susceptible to noise than single-point-based representations. To that end, we propose a point-to-distribution transformer based on variational autoencoder. The transformer merges visual and semantic features to learn an informative latent space. Our DFSL achieves state-of-the-art results on four popular few-shot benchmarks, demonstrating its effectiveness and innovation in addressing the challenges posed by few-shot learning.

## Download the Datasets
1. Download these zipped files and put them into './data/dataIm'.
* [miniImageNet](https://drive.google.com/file/d/1g4wOa0FpWalffXJMN2IZw0K2TM2uxzbk/view) 
* [tieredImageNet](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)
* [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
* [CUB](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)
* [glove word embedding](https://nlp.stanford.edu/projects/glove/)

2. (Optional) Download pretrained checkpoints [here](https://drive.google.com/drive/folders/1unnbnYgjXtwP4lFtcLrCAcZ_H1uQESLf?usp=share_link) and extract features to './data/features'.


## Running Experiments
If you want to train the models from scratch, please run the run_pre.py first to pretrain the backbone. Then specify the path of the pretrained checkpoints to "./checkpoints/[dataname]"
* Run pretrain phase:
```bash
python run_pre.py
```
* Run fsl train and test phase:
```bash
python run_dfsl.py
```
## LISENCE
* All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode

* The license gives permission for academic use only.

## Acknowledgments
Our project references the codes in the following repos.

* [**POODLE: Improving Few-shot Learning via Penalizing Out-of-Distribution Samples**](https://github.com/lehduong/poodle)
* [**PyTorch VAE**](https://github.com/AntixK/PyTorch-VAE)
