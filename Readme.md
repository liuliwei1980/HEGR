# herg

Welcome to use a new multi-type feature fusion framework MTF-hERG to accurately predict the cardiotoxicity of hERG compounds. This framework integrates molecular features such as molecular fingerprints, 2D molecular images and 3D molecular diagrams to fully capture the internal structure and properties of compounds. The classification prediction of whether the compound is a hERG blocker and the regression prediction of its hERG inhibitory ability are realized.

# **Command**

-python 3.8.10
-torch 1.11.0
-numpy 1.22.4
-rdkit 2023.9.6
-torch-geometric 2.5.3                    
-torch-scatter 2.0.9                    
-torchvision 0.12.0+cu113  

### ** Train**
If you want to run the classification calculation, you don't need any changes, just run the main.py file.

If you want to run regression calculation, please change the data set to hergIC.csv data set in the readData file.


## Citation

If you find this repo useful, please cite our paper

## Contact
If you have any question, please contact us: w635312612@163.com# HEGR
