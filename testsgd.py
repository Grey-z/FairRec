"""
The implementation of TestSGD for recommender systems.
"""
import pandas as pd
import numpy as np
import sys
import os
import math
import random
import time
import tensorflow as tf
from utilis import get_rule_list, create_single_rule, create_sparse_rule_query
from utilis import get_fairness_score
from utilis import get_test_data, get_saved_model_details, get_model


pd.set_option('mode.chained_assignment', None)
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def testsgd(model_info,test_info):
    """
    The implementation of Themis for recommender systems.

    Args:
    model_info: dict, including test_model and train_features.
        model_info['test_model']: model, recommendation model to be tested.
        model_info['train_features']:list, features for model training.
    test_info：dict, including dataset, test_data, sensitive_features,metric and threshold.
        test_info['dataset']:str, the dataset to be tested.
        test_info['test_data']: pandas, data containing information about the users to be tested.
        test_info['sensitive_features']: list, sensitive attributes to be tested.
        test_info['metric']: str, metric used in the fairness testing.
        test_info['threshold']:float, criteria for filtering groups.

    Return:
    uf_score: float, the result of unfairness score.
    time_themis: float, time consumed for the testing.
    """
    start_time = time.time()
    save_result = []
    save_group_score = []
    rule_value_list = []
    sensitive_features = test_info['sensitive_features']
    test_data = test_info['test_data']
    if test_info['dataset'] == 'last-fm':
        age_info = test_data['age'].tolist()
        age_list = [x//10 for x in age_info]
        test_data['age_level'] = age_list
        sensitive_features = ['gender','age_level','country']
    threshold = test_data['user_id'].nunique() * test_info['threshold']
    metric = test_info['metric']
    
    average_score = get_fairness_score(model_info,test_data,metric)
    nums_all = test_data['user_id'].nunique()
    for feats in sensitive_features:
        max_value = test_data[feats].max()
        feat_list = range(max_value + 1)
        rule_value_list.append(feat_list)
    rule_list = get_rule_list(rule_value_list, 0, [], [])

    for value in rule_list:
        rule = create_single_rule(sensitive_features,value)
        rule_query = create_sparse_rule_query(rule)
        group_target = test_data.query(rule_query)
        group_nums = group_target['user_id'].nunique()

        if (group_target['user_id'].nunique()) >=  threshold:
            score_target = get_fairness_score(model_info,group_target,metric)
            save_group_score.append(score_target)
            score_compare = (average_score * nums_all - score_target * group_nums) / (nums_all - group_nums)
            metric_result = score_target - score_compare
            save_result.append(metric_result)
    max_index = save_result.index(max(save_result))
    min_index = save_result.index(min(save_result))
    uf_score = save_group_score[max_index] - save_group_score[min_index]
    end_time = time.time()
    time_testsgd = end_time-start_time
    return uf_score, time_testsgd
