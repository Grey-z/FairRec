import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Data process')
parser.add_argument('--path',required=True, type=str, help='data path or a dir' )

def create_last_fm_data(data):
    """
    Create train set and test set.
    
    Args:
    data: pandas, origin data.
        
    Return:
    train_pd: pandas, data for model training.
    test_pd: pandas, data for testing.
    """
    
    # data = data.sort_values(['user_id','timestamp'])
    num = len(data.user_id.value_counts())
    test_pd = pd.DataFrame(columns=data.columns)
    for user_id, hist in tqdm(data.groupby('user_id')):
        user_select_group = hist.loc[hist['user_id'] == user_id]
        user_pd = user_select_group.iloc[-1]
        test_pd = test_pd.append(user_pd)
    drop_index = test_pd.index.to_list()
    train_pd = data[~data.index.isin(drop_index)]
    return train_pd,test_pd

def neg_sample(u_data, neg_rate_train=1,neg_rate_test = 19):
    """
    Sampling items that users have not interacted with as negative samples for training set and test set.
    
    Args:
    u_data: pandas, the record data of users.
    neg_rate_train: int, the ratio of negative samples and positive samples in the training set.
    neg_rate_test: int, the ratio of negative samples and positive samples in the test set.
    
    Return:
    u_data_neg_train: pandas, negative samples in the training set.
    u_data_neg_test: pandas, negative samples in the test set.
    """
    item_ids = u_data['item_id'].unique()
    print('start neg sample')
    neg_data_train = []
    neg_data_test = []
    for user_id, hist in tqdm(u_data.groupby('user_id')):
        rated_list = hist['item_id'].tolist()
        candidate_set = list(set(item_ids) - set(rated_list))
        neg_list_train = np.random.choice(candidate_set, size=(len(rated_list)-1) * neg_rate_train, replace=True)
        neg_list_test = np.random.choice(candidate_set, size=neg_rate_test, replace=True)
        for id in neg_list_train:
            neg_data_train.append([user_id, id, 0])
        for id in neg_list_test:
            neg_data_test.append([user_id, id,  0])
    u_data_neg_train = pd.DataFrame(neg_data_train)
    u_data_neg_test = pd.DataFrame(neg_data_test)
    u_data_neg_train.columns = ['user_id', 'item_id', 'label']
    u_data_neg_test.columns = ['user_id', 'item_id', 'label']
    return u_data_neg_train,u_data_neg_test

if __name__ == '__main__':
    opt = parser.parse_args()
    data_path = opt.path
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # load data of user_file.
    user_file = data_path + 'usersha1-profile.tsv'
    user_data = pd.read_csv(user_file, sep='\t', names=['user_id','gender','age','conutry','signup_date'])
     
    # load data of records.
    row_file = data_path + 'usersha1-artmbid-artname-plays.tsv'
    row_data = pd.read_csv(row_file, sep='\t',names = ['user_id','item_id','cate','plays'])
    
     # delete data with vacant values.
    user_data = user_data.dropna(axis=0, how='any')
    preserve_data = row_data.loc[row_data['user_id'].isin(user_data['user_id'])]
    user_data = user_data.loc[user_data['user_id'].isin(preserve_data['user_id'])]
    
    # label encoding.
    lbe_feats = ['user_id','item_id','cate']
    for feat in lbe_feats:
        lbe = LabelEncoder()
        preserve_data[feat] = lbe.fit_transform(preserve_data[feat])
    
    lbe_feats_user = ['user_id','gender','age','country','signup_date']
    for feat in lbe_feats_user:
        lbe_user = LabelEncoder()
        user_data[feat] = lbe_user.fit_transform(user_data[feat])
    
    # split dataset for train_data and test_data.
    train_data , test_data = create_last_fm_data(preserve_data)
    
    # get negative samples for training set and test set.
    u_data_neg_train,u_data_neg_test = neg_sample(preserve_data, neg_rate_train=1,neg_rate_test=49)
    item_info = preserve_data.loc[:,['item_id','cate']]
    item_info = item_info.drop_duplicates(['item_id'])
    u_data_neg_train = pd.merge(neg_train, item_info , on=['item_id'],how = 'left')
    u_data_neg_test = pd.merge(neg_test, item_info , on=['item_id'],how = 'left')
    
    # get training set.
    train_data = train_data.loc[:,['user_id','item_id','cate']]
    train_data['label'] = 1
    train_set = pd.concat([train_data, u_data_neg_train])
    train_set = pd.merge(train_set, user_data , on=['user_id'],how = 'left')
    
    # save training set
    train_set.to_csv('./train_data.csv')
    
    # get test set.
    test_data = test_data.loc[:,['user_id','item_id','cate']]
    test_data['label'] = 1
    test_set = pd.concat([test_data, u_data_neg_test])
    test_set = pd.merge(test_set, user_data , on=['user_id'],how = 'left')
    
    # calculate the popularity and user_pop.
    popularity_list = []
    for i in item_info['item_id'].to_list():
        select_item = preserve_data.loc[preserve_data['item_id'] == i]
        popularity_list.append(len(select_item) / len(preserve_data) * 100)
    item_info['popularity'] = popularity_list
    item_info = item_info.loc[:,['item_id','popularity']]
    test_set = pd.merge(test_set, item_info , on=['item_id'],how = 'left')
    result = []
    for user_id, hist in tqdm(train_data.groupby('user_id')):
        user_select_group = hist.loc[hist['user_id'] == user_id]
        user_list = user_select_group['popularity'].tolist()
        user_pop = np.mean(user_list)
        result.append([user_id,user_pop])
    user_pop_pd = pd.DataFrame(result,columns=['user_id','user_pop'])
    test_data = pd.merge(test_data, user_pop_pd,  on=['user_id'],how = 'left')
    
    # save test set.
    test_data.to_csv('./test_data_liuyi.csv')
    