# Introduction 

This repository contains code for the paper **FairRec: Fairness Testing for Deep Recommender Systems**, which has been accepted by ISSTA 2023.
Pre-printed versions of the paper are available from https://arxiv.org/abs/2304.07030
# Instruction
1. Install the related environment configuration and dependency packages.
2. Prepare datasets and models to be tested.
3. Modify the config.json.

- To select the datasets for testing, modify the data_params/dataset.(Note: 'ml-1m' denotes MovieLens-1M, 'last-fm' denotes LFM360K, 'black-fri' denotes BlackFriday and 'amazon' denotes Amazon).

- To select the model for testing, modify the model_params/model_name.(Note: 'wdl' denotes Wide&Deep, 'deepfm' denotes 'DeepFM', 'dcn' denotes 'DCN' and 'fgcnn' denotes FGCNN).

- Select the metric for testing, modify the test_info/metric. (Note: 'auc' denotes AUC, 'mrr' denotes 'MRR', 'ndcg' denotes 'NDCG', 'diversity' denotes 'URD' and popularity denotes 'URP').

- Select sensitive features you interested for testing, modify test_info/sensitive_features.

- Modify the params for dpso. (Note: For small dataset (with group less than 1000), we recommend that the initial number of particles be no less than 50 ).

4. run the run_fairrec.py.

You can also use FairRec to test your own datasets and models using any metrics.
1. Refer to data_process_lfm360k.py to process the new dataset.
2. Refer to model_train.py to train the new models.
3. Modify the utilis.py to load your own datasets and models.
4. Modify the rec_metrics.py to add the new metrics for testing.
5. Modify the fairrec.py/def get_score() to use the new metrics.



# Environment
This project is based on **Python 3.8.1** and **Tensorflow 2.3.0**.

Packages:
```
deepctr==0.9.3
numpy==1.18.5
pandas==1.3.2
tensorflow==2.3.0
```

# Files

- **model** : The trained model (Wide&Deep, DeepFM, DCN and FGCNN) based on four datasets.
- **config.json**: File for configuration parameters.
- **data_process_lfm360k.py**: An examples of data processing for LFM360K.
- **example_testsgd.py** : An example for testsgd.
- **example_themis.py** : An example for themis.
- **fairrec.py** : Our FairRec testing framework.
- **model_train.py**: An examples of model training for DeepFM on LFM360K.
- **rec_metrics.py** : The metrics used for testing.
- **run.sh**: Command to run FairRec.
- **run_fairrec.py** :  An example for FairRec.
- **themis.py** : An implementation of [1] for recommender system.
- **testsgd.py** : An implementation of [2] for recommender system.
- **utilis.py** : General purpose functions.

## Reference:
[1] Sainyam Galhotra, Yuriy Brun, and Alexandra Meliou. 2017. Fairness testing: testing software for discrimination. In Proceedings of the 2017 11th Joint meeting on foundations of software engineering. 498–510

[2] Mengdi Zhang, Jun Sun, Jingyi Wang, and Bing Sun. 2022. TESTSGD: Interpretable Testing of Neural Networks Against Subtle Group Discrimination. arXiv preprint arXiv:2208.11321 (2022)

[3] Weichen Shen. (2017). DeepCTR: Easy-to-use,Modular and Extendible package of deep-learning based CTR models. https://github.com/shenweichen/deepctr.

# Data 
Subject to file size limitations, the processed data used in this work can be downloaded from the following links, including four datasets, MovieLens, BlackFriday, LFM360K and Amazon.
https://drive.google.com/drive/folders/1fzSaUSqns1tRKG_atsu_rxwFvMe1F4XQ?usp=sharing
## Data Format
![data](https://user-images.githubusercontent.com/117908227/202380517-a7638d84-ba6d-415b-9f2e-a17dc345b2c1.jpg)

- user_id : The id of each user.
- item_id : The id of each item.
- gender/age/country : The sensitive attributes.
- cate : Required to calculate the metric of $UF_{urd}$.
- popularity/user_pop: Required to calculate the metric of $UF_{urp}$

