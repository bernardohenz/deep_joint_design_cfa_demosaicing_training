# Deep Joint Design of Color Filter Arrays and Demosaicing

This is the training code using Keras (version > 2.0) for the article Deep Joint Design of Color Filter Arrays and Demosaicing.

## Prerequisites
- Python 2 or 3
- Keras and Tensorflow


## Getting Started

### Installation
- Install python packages by using
```bash
pip install -r requirements.txt
```
-If you want to run on a GPU, just uninstall tensorflow e install its GPU version:
```bash
pip uninstall tensorflow
pip install tensorflow-gpu==1.15
```

### Running our models in images
- This training code was updated for the newer version of Keras and Tensorflow and is not the same used in the original paper. For the results reported in the paper, please follow to our [test repository](https://github.com/bernardohenz/deep_joint_design_cfa_demosaicing "Test Repository"), which includes our trained models and script for reconstruction.

- For a description of the parameters, please type:
```python
python train.py -h
```

## Citation
If you use this code, please cite our paper
```
@article{HenzGastalOliveira_2018,
    author = {Bernardo Henz and Eduardo S. L. Gastal and Manuel M. Oliveira},
    title   = {Deep Joint Design of Color Filter Arrays and Demosaicing},
    journal = {Computer Graphics Forum},
    volume = {37},
    year    = {2018},
    }
```
