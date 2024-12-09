# LoRATSF: Modeling Long-term Time Series Forecasting with *1k* Parameters using LoRA

**LoRATSF** (Lightweight Long-term Time Series Forecasting) is my final project for the Python course in my **second year of Nankai University**. 

This project is based on the previous [**SparseTSF** model](https://arxiv.org/pdf/2405.00946) (LSS et al., 2024), 
with optimizations and improvements aimed at further reducing the model's parameter count, 
focusing purely on minimizing the number of parameters without compromising the model's predictive power. 


## Related Work

**LoRATSF** builds upon the **SparseTSF** model, which introduced a novel approach to time series forecasting, featuring:

- **Lightweight design**: The model has **100** parameters, making it orders of magnitude smaller than other models.
- **Decoupling of periodicity and trends**: Through **Cross-Period Sparse Forecasting**, LoRATSF effectively separates periodic and trend components in time series data like SparseTSF.
- **Excellent generalization capability**: LoRATSF has shown outstanding cross-domain generalization ability across multiple datasets, especially in scenarios with limited computational resources or smaller datasets like SparseTSF.

The reference for **SparseTSF** is as follows:

```bibtex
@inproceedings{SparseTSF,
  author    = {LSS, et al.},
  title     = {SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters},
  booktitle = {ICML 2024},
  year      = {2024},
  url       = {https://arxiv.org/pdf/2405.00946},
}
```

## Project Background

This project is my final assignment for the Python course, aimed at exploring some probabilities to build an efficient long-term time series forecasting model. 
The advantages of this model are:

*   **Few parameters**: Making it deployable in resource-constrained environments.

## Project Objectives

1.  **Optimize the SparseTSF model** to minimize its parameters.
2.  **Demonstrate the model’s performance** on multiple time series datasets, including traffic data, electricity load, etc.

## Environment Setup

Ensure that you have Conda installed on your system, and then configure your environment as follows:

```bash
conda create -n LoRATSF python=3.8
conda activate LoRATSF
pip install -r requirements.txt
```

## Data Preparation

All necessary time series datasets can be found at the following link (provided by the original **SparseTSF** project):

[Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

Create a folder `./dataset` and place all CSV files inside it, for example: `./dataset/ETTh1.csv`.

## Acknowledgement

Thanks to the original authors of SparseTSF (LSS et al.), whose work provided invaluable insights for this project. Also, thanks to ICML 2024 for accepting our submission, allowing this work to be shared with the community.

