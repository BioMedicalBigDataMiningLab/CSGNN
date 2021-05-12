# CSGNN
The code for paper "CSGNN: Contrastive Self-Supervised Graph Neural Network for Molecular Interaction Prediction"

![overview](.//overview.jpg)

## 1. Overview

The repository is organized as follows:

+ `data/` contains the datasets used in the paper;
+ `parms_setting.py` contains hyperparameters adopted by CSGNN;
+ `data_preprocess.py` is the preprocess of data before training;
+ `layer.py` contains mix-hop GNN layers and contrastive GNN layers;
+ `instantiation.py` instantiates the CSGNN;
+ `train.py` contains the training and testing code on datasets;
+ `utils.py` contains preprocessing functions of the data (e.g., normalize...);
+ `main.py` contains entry to CSGNN (e.g., normalize...);


## 2. Dependencies
* numpy == 1.18.5
* scipy == 1.5.2
* sklearn == 0.23.2
* torch == 1.5.0
* torch-geometric == 1.6.1
* networkx == 2.4


## 3. Example
Here we provide several example of using CSGNN:
To run CSGNN with GCN decoder on DTI network using "uniform" as initial features and output the result to test.txt, execute the following command:

```shell
python main.py --aggregator GCN --feature_type uniform --in_file data/DTI.edgelist --out_file test.txt
```



To run CSGNN with GIN decoder on DDI network using "position" as initial features and output the result to test.txt, execute the following command:

```shell
python main.py --aggregator GIN --feature_type position --in_file data/DDI.edgelist --out_file test.txt
```

