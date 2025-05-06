# Group-specific discriminant analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shuo-zhou/GSDA-Lateralization/blob/main/gsda_demo.ipynb)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/shuo-zhou/GSDA-Lateralization/blob/main/README.md)
[![DOI](https://zenodo.org/badge/325192453.svg)](https://zenodo.org/doi/10.5281/zenodo.13626594)

## Introduction

This repository contains the implementation of Group-Specific Discriminant Analysis (GSDA) and experiments from the GigaScience paper “Group-specific discriminant analysis enhances detection of sex differences in brain functional network lateralization” by Zhou et al. (2025).

## Framework

![GSDA](figures/GSDA.png)

## Datasets

The resting-state fMRI data from [HCP](https://www.humanconnectome.org) [[1](#references)] and [GSP](https://www.neuroinfo.org/gsp/) [[2](#references)] is used in this study. Code for data preprocessing is available at `/preprocess`. Processed data is available at Zenodo: [[HCP](https://doi.org/10.5281/zenodo.10050233)], [[GSP](https://doi.org/10.5281/zenodo.10050234)].

## System Requirements

```(text)
numpy>=1.24.3
pandas>=1.5.3
scipy>=1.10.1
scikit-learn>=1.2.2
pytorch>=2.0.0
yacs
```

## Installation Guide

```(bash)
pip install -r requirements.txt
```

## Instructions for Use

Basic usage:

```(bash)
python main.py --cfg configs/demo-hcp.yaml
```

Please create more .yaml files for different random seeds and datasets.

## Demo

We provide GSDA running demo through a cloud Jupyter notebook on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shuo-zhou/GSDA-Lateralization/blob/main/gsda_demo.ipynb). Note the number of repetition is limited for faster demonstrations. This demo takes 10-20 minutes to complete the training and testing process.

## References

[1] Smith, S. M. et al. Resting-state fMRI in the human connectome project. _NeuroImage_ 80, 144–168 (2013)

[2] Holmes, A. J. et al. Brain genomics superstruct project initial data release with structural, functional, and behavioral measures. _Sci. Data_ 2, 1–16 (2015)

## Citation

If you use this code in your research, please cite the following paper:

```(text)
@article{zhou2025group,
  title={Group-specific discriminant analysis enhances detection of sex differences in brain functional network lateralization},
  author={Zhou, Shuo and Luo, Junhao and Jiang, Yaya and Wang, Haolin and Lu, Haiping and Gaolang, Gong},
  journal={GigaScience},
  year={2025},
  publisher={Oxford University Press}
}
```
