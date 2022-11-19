import pandas as pd
import numpy as np
import sys
import os
import math
import json
import argparse
import tensorflow as tf
from utilis import get_test_data, get_saved_model_details, get_model
from fairrec import FairRec, set_dpso_params, run_fair_rec


parser = argparse.ArgumentParser(description='Testing for FairRec')
parser.add_argument('--config',required=True, type=str, help='config file' )

pd.set_option('mode.chained_assignment', None)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    opt = parser.parse_args()
    config_file = opt.config
    
    # load params
    with open(config_file, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    fairrec_params = dicts['fairrec']
    model_params = fairrec_params['model_params']
    data_params = fairrec_params['data_params']
    
    # load data.
    test_data = get_test_data(model_params,data_params)
    
    # load model.
    check_path ,train_features = get_saved_model_details(model_params, data_params)
    model = get_model(model_params,data_params)
    
    # set params.
    test_info = fairrec_params['test_info']
    test_info['test_data'] = test_data
    test_info['test_model'] = model
    test_info['train_features'] = train_features
    
    dpso_params = fairrec_params['dpso_params']
    
    # run fair_rec.
    run_nums = 5
    time_list = []
    unfairness_list = []
    for i in range(run_nums):
        uf_score, test_time = run_fair_rec(dpso_params,test_info)
        time_list.append(test_time)
        unfairness_list.append(uf_score)
    mean_uf_score = np.mean(unfairness_list)
    mean_time = np.mean(time_list)
    
    # print result
    print('mean unfairness socre: {}'.format(mean_uf_score))
    print('mean time consumption: {}'.format(mean_time))