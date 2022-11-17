# Introduction 

This repository contains code for the paper **FairRec: Fairness Testing for Deep Recommender Systems**

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

- **data** : The processed data of four datasets (MovieLens, LFM360K, BlackFriday and Amazon).
- **model** : The trained model (Wide&Deep, DeepFM, DCN and FGCNN) based on four datasets.
- **fair_rec** : Our FairRec testing framework,
- **baseline** : An implementation of [1] and [2] for recommender system.

## Reference:
[1] Sainyam Galhotra, Yuriy Brun, and Alexandra Meliou. 2017. Fairness testing: testing software for discrimination. In Proceedings of the 2017 11th Joint meeting on foundations of software engineering. 498–510

[2] Mengdi Zhang, Jun Sun, Jingyi Wang, and Bing Sun. 2022. TESTSGD: Interpretable Testing of Neural Networks Against Subtle Group Discrimination. arXiv preprint arXiv:2208.11321 (2022)

[3] Weichen Shen. (2017). DeepCTR: Easy-to-use,Modular and Extendible package of deep-learning based CTR models. https://github.com/shenweichen/deepctr.

# Data Format

![data](https://user-images.githubusercontent.com/117908227/202380517-a7638d84-ba6d-415b-9f2e-a17dc345b2c1.jpg)

- user_id : The id of each user.
- item_id : The id of each item.
- gender/age/country : The sensitive attributes.
- cate : Required to calculate the metric of $UF_{urd}$.
- popularity/user_pop: Required to calculate the metric of $UF_{urp}$
