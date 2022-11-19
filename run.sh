#!/usr/bin/env bash

# data process
python data_process_lfm360k.py --path='./data/lfm360k/'

# train model
python model_train.py --data_path = './data/last-fm/processed_data/train_data.csv' --model_path='./model/' --model_name='deepfm' --batch_size=256 --epochs=10

# FairRec
python run_example.py --config='config.json'