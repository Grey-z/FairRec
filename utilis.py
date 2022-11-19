import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import time
import os
from rec_metrics import RecMetrics
import deepctr
from deepctr.models import *
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names


def get_test_data(model_params,data_params):
    """
    Load data for testing.
    
    Args:
    model_params: dict, including model_name,embedding_dim and batch_size.
        model_params['model_name']:str, select the model to be tested.
            Note: Wide&Deep: wdl , DeepFM: deepfm , DCN: dcn, FGCNN: fgcnn.
        model_params['embedding_dim']: int, the params of embedding_dim for models.
        model_params['batch_size']: int, the params of batch_size for models.
    data_params: dict, including dataset.
        data_params['dataset']: str, select the dataset to be tested.
        Note: MovieLens: ml-1m, LFM360K: last-fm, BlackFriday: black-fri, Amazon: amazon
        
    Return:
    test_data:pandas, test data of selected dataset.
    """
    
    dataset = data_params['dataset']
    data_path = data_params['data_dir']
    test_data_dir = data_path + dataset + '/processed_data/'
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    test_data = pd.read_csv(test_data_dir + 'test_data.csv',index_col=0)
    if dataset == 'ml-1m':
        cate_list = [eval(s) for s in test_data['cate']]
        test_data['cate'] = cate_list
    return test_data

def get_saved_model_details(model_params, data_params):
    """
    Load path and training features for saved models.
    
    Args:
    model_params: dict, including model_name,embedding_dim and batch_size.
        model_params['model_name']:str, select the model to be tested.
            Note: Wide&Deep: wdl , DeepFM: deepfm , DCN: dcn, FGCNN: fgcnn.
        model_params['embedding_dim']: int, the params of embedding_dim for models.
        model_params['batch_size']: int, the params of batch_size for models.
    data_params: dict, including dataset.
        data_params['dataset']: str, select the dataset to be tested.
        Note: MovieLens: ml-1m, LFM360K: last-fm, BlackFriday: black-fri, Amazon: amazon
        
    Return:
    check_path: str, the path where the model is saved.
    train_feats: list, the features for model training.
    """
    model_name = model_params['model_name']
    dataset = data_params['dataset']
    model_dir = model_params['model_dir']
    string =  model_name + '-' + dataset 
    saved_model_dir = model_dir + string + '/'
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    model_file = string + '.ckpt'
    check_path = saved_model_dir + model_file
    if dataset == 'ml-1m':
        train_feats = ["item_id", "user_id", "gender", "age", "occupation", "zipcode"]
    if dataset == 'black-fri':
        train_feats = ['user_id','item_id','gender','age','occupation','city','stay_years','martial_status','cate','subcate_1','subcate_2','price']
    if dataset == 'last-fm':
        train_feats = ['user_id','item_id','gender','age','country','signup_date','cate']
    if dataset == 'amazon':
        train_feats = ['user_id','item_id','price','cate']
    return check_path , train_feats

def get_feat_dict(dataset):
    """
    Load details for selected dataset.
    
    Args:
    dataset: str, select the dataset to be tested.
        Note: MovieLens: ml-1m, LFM360K: last-fm, BlackFriday: black-fri, Amazon: amazon
        
    Return:
    feat_dict: dict, including the number of encodings for each training feature.
    """
    
    if dataset == 'ml-1m':
        feat_dict = {'item_id': 3706,
                 'user_id': 6040,
                 'gender': 2,
                 'age': 7,
                 'occupation': 21,
                 'zipcode': 3439}
    if dataset == 'black-fri':
        feat_dict = {'user_id': 5891,
                 'item_id': 3631,
                 'gender': 2,
                 'age': 7,
                 'occupation': 21,
                 'city': 3,
                 'stay_years': 5,
                 'martial_status': 2,
                 'cate': 20,
                 'subcate_1': 18,
                 'subcate_2': 16,
                 'price': 23962}
    if dataset == 'last-fm':
        feat_dict = {'user_id': 267034,
                     'item_id': 145975,
                     'gender': 2,
                     'age': 111,
                     'country': 239,
                     'signup_date': 2254,
                     'cate': 249841}
    if dataset == 'amazon':
        feat_dict = {'user_id': 188657,
                     'item_id': 29391,
                     'price': 1001,
                     'cate': 3516}
    return feat_dict

def get_model(model_params,data_params):
    
    """
    Load selected models.
    
    Args:
    model_params: dict, including model_name,embedding_dim and batch_size.
        model_params['model_name']:str, select the model to be tested.
            Note: Wide&Deep: wdl , DeepFM: deepfm , DCN: dcn, FGCNN: fgcnn.
        model_params['embedding_dim']: int, the params of embedding_dim for models.
        model_params['batch_size']: int, the params of batch_size for models.
    data_params: dict, including dataset.
        data_params['dataset']: str, select the dataset to be tested.
        Note: MovieLens: ml-1m, LFM360K: last-fm, BlackFriday: black-fri, Amazon: amazon.
        
    Return:
    model: model, the model to be tested.
    """
    
    dataset = data_params['dataset']

    model_name = model_params['model_name']
    embedding_dim = model_params['embedding_dim']
    check_path, train_feats = get_saved_model_details(model_params,data_params)
    
    feat_dict = get_feat_dict(dataset)
    fixlen_feature_columns = [SparseFeat(feat, feat_dict[feat], embedding_dim=embedding_dim)
                                      for feat in train_feats]
    linear_feature_columns = fixlen_feature_columns 
    dnn_feature_columns = fixlen_feature_columns 
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    if model_name == 'deepfm':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
        print('load DeepFM model')
    if model_name == 'wdl':
        model = WDL(linear_feature_columns, dnn_feature_columns, task='binary')
        print('load WDL model')
    if model_name == 'dcn':
        model = DCN(linear_feature_columns, dnn_feature_columns, task='binary')
    if model_name == 'fgcnn':
        model = model = FGCNN(linear_feature_columns, dnn_feature_columns, conv_kernel_width=(3, 2), conv_filters=(2, 1), new_maps=(
        2, 2), pooling_width=(2, 2), dnn_hidden_units=(32,), dnn_dropout=0.5, )
    # model.summary()
    model.load_weights(check_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',
                  metrics=['AUC', 'Precision', 'Recall'])
    return model

def get_fairness_score(model_info,group_target,metric):
    
    """
    Calculate the metric socre for the selcted group.

    Args:
    model_info: dict, including test_model and train_features.
        model_info['test_model']: model, recommendation model to be tested.
        model_info['train_features']:list, features for model training.
    
    group_target: pandas, data containing information about the users to be tested.
    metric: str, metric used in the fairness testing.

    Return:
    score_target: float, the result of metric score.
    """
    
    train_features = model_info['train_features']
    model = model_info['test_model']
    test_input = {name: group_target[name].values for name in train_features}
    y_pred = model.predict(test_input)
    group_target['predict'] = np.array(y_pred)
    
    if metric == 'mrr':
        score_target = RecMetrics(group_target).mrr_score()
    if metric == 'auc':
        score_target = RecMetrics(group_target).auc_score()
    if metric == 'ndcg':
        score_target = RecMetrics(group_target).ndcg_score()
    if metric == 'urd':
        score_target = RecMetrics(group_target).cate_diversity_score()
    if metric == 'urp':
        score_target = RecMetrics(group_target).popularity_score()
    
    return score_target

def create_single_rule(feats,value):
    """
    Create a single rule for selecting group.
    
    Args:
    feats: list, sensitive attributes selected to be tested. e.g. ['gender', 'age'].
    value: list, the specific values of each sensitive attribute. e.g. [0, 18].
     
    Return:
    rule: dict, including the specific values of each sensitive attribute. e.g. {'gender':0, 'age': 18}.
    """
    
    rule = {}
    for i in range(len(feats)):
        rule[feats[i]] = value[i]
    return rule


def create_sparse_rule_query(rule):
    
    """
    Create the query to select users who meet specified requirements.
    
    Args:
    rule: including the specific values of each sensitive attribute. e.g. {'gender':0, 'age': 18}.
        
    Return:
    rule_query: str, the query to select users who meet specified requirements. e.g. '(gender == 0) & (age == 18)'.
    """
    
    rule_query = ''
    key = rule.keys()
    length = len(key)
    count = 0
    for i in key:
        count = count + 1
        rule_query = rule_query + '( ' + i +' == ' + str(rule[i]) + ')'
        if count < length:
            rule_query = rule_query + ' & '
    return rule_query


def even(x):
    """
    Check if a value is less than -99.
    
    Args:
    x: float, a number.
    
    Return
    bool, True or False.
    """
    
    return x < -99


def get_rule_list(data, cur_y_idx = 0, lst_rst = [], lst_tmp = []):
    """
    Get the combination of all sensitive attributes.
    
    Args:
    data: list, the range of all sensitive attributes. e.g.[range(2),range(9)].
    cur_y_idx: int, 0.
    lst_rst: list, an empty list.
    lst_tmp: list, an empty list.
    
    Return
    lst_rst, the combination of all sensitive attributes. e.g. [[0,0],[0,1],..., [1,8]].
    """
    max_y_idx = len(data) - 1  
    for x_idx in range(len(data[cur_y_idx])):  
        lst_tmp.append(data[cur_y_idx][x_idx])  
        if cur_y_idx == max_y_idx:  
            lst_rst.append([*lst_tmp])
        else:       
            get_rule_list(data, cur_y_idx+1, lst_rst, lst_tmp)
        lst_tmp.pop()
    return lst_rst

