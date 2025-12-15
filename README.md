# PointONet: Predicting physical fields using a PointNet-based neural operator with arbitrary sensors
Sample codes for a PointNet-based neural operator learning framework (PointONet) that integrates arbitrary sensors to accurately predict physical fields. In the model architecture, the branch network integrates the permutation invariance and geometric feature extraction of PointNet to adaptively encode arbitrary sensors, where the PointNet-based architectures is designed to characterize varying input function. Meanwhile, the trunk network encodes the query coordinates of the output function. By combing the branch and trunk network, the PointONet approximates operator relationships between function spaces.

# Workflow
<div align="center">
  <img src="https://github.com/liuxu97531/PointONet/blob/main/figs/models.PNG?raw=true" alt="Main Result" style="width:50%;">
  <p><em>Figure 1: Overview of our DSPO framework.</em></p>
</div>


## Information
```python
├── data/ # raw data and the data generator
│ ├── ODE_1d/ # the data generator
│ │ └── _dataset.py 
│ ├── DRE_2d/
│ │ └── _dataset.py 
│ ├── Burger_2d/ 
│ │ └── _dataset.py 
│ └── Bracket_3d # raw data
├── experiments/ # Experiment 
│ ├── ODE_1d/
│ │ └── _main.py 
│ ├── DRE_2d/
│ │ └── _main.py
│ ├── Burger_2d/ 
│ │ └── _main.py
│ └── Bracket_3d/ 
│ │ └── _main.py
├── model/ # Model definitions, network structures, model configurations
│ ├── deepOnet/ # Baseline model
│ ├── pointnet/ # Baseline model
│ └── pointOnet/ # Proposed method
├── utils/ # Common utility functions and helper modules
└── README.md # Project overview and instructions
```
The raw data (`VolumeMesh, xyzdmlc.npz and target.npz`) for Bracket_3d can be downloaded in https://www.kaggle.com/datasets/jangseop/point-deeponet-dataset.

## Usage
We can run the code by revising `cases_name` with `ODE_1d`, `DRE_2d`, `Burger_2d`, `Bracket_3d`. 
```python
python /experiments/cases_name/main.py
```

## Requirements
```
conda create -n jax_env python=3.9 -y
conda activate jax_env
pip install jax jaxlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Reference
Yuxiang Weng, Xu Liu, Jiansha Lua, Wei Peng, Wen Yao. PointONet: Predicting physical fields using a PointNet-based neural operator with arbitrary sensors, arXiv preprint arXiv.