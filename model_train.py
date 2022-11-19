import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tqdm import tqdm
import deepctr
from deepctr.models import *
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from utilis import get_feat_dict


parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--data_path',required=True, type=str, help='data path or a dir')
parser.add_argument('--model_path',required=True, type=str, help='path or a dir for saving model')
parser.add_argument('--model_name',required=True, type=str, help='selected model for training')
parser.add_argument('--batch_size',required=True, type=str, help='batch_size')
parser.add_argument('--epochs',required=True, type=str, help='epochs')


def train_ctr_model(model_name,dataset,model_path,batch_size, epochs,validation_split=0.2):
    """
    Training model.
    
    Args:
    model_name: str, the selected model.
    dataset: str, the selected dataset.
    model_path: str, the path to save the model.
    batch_size: int, the params batch_size for model training.
    epochs: int, the params epochs for model training.
    
    """
    if model_name == 'deepfm':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    if model_name == 'wdl':
        model = WDL(linear_feature_columns, dnn_feature_columns, task='binary')
    if model_name == 'dcn':
        model = DCN(linear_feature_columns, dnn_feature_columns, task='binary')
    if model_name == 'fgcnn':
        model = model = FGCNN(linear_feature_columns, dnn_feature_columns, conv_kernel_width=(3, 2), conv_filters=(2, 1),
                              new_maps=(2, 2), pooling_width=(2, 2), dnn_hidden_units=(32,), dnn_dropout=0.5)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    saved_model_path = model_path + model_name + '-' + dataset + '/'
    check_path = model_path + model_name + '-' + dataset + '-liuyi.ckpt'
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',
              metrics=['AUC', 'Precision', 'Recall'])
    model.summary()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
    print('begain training')
    model.fit(train_model_input, label,
						batch_size=batch_size, epochs=epochs, verbose=2,
                        callbacks=[checkpoint],
						validation_split=validation_split)

if __name__ == '__main__':
    opt = parser.parse_args()
    data_path = opt.data_path
    model_path = opt.model_path
    model_name = opt.model_name
    batch_size = opt.batch_size
    epochs = opt.epochs
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train_data = pd.read_csv(data_path,index_col=0)
    
    # get fixlen_feature_columns and model input
    dataset = 'last-fm'
    feat_dict = get_feat_dict(dataset)
    train_feats = ['user_id','item_id','gender','age','country','signup_date','cate']
    fixlen_feature_columns = [SparseFeat(feat, feat_dict[feat], embedding_dim=4)
                                      for feat in train_feats]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train_data = sklearn.utils.shuffle(train_data)
    train_model_input = {name: train_data[name].values for name in train_feats}
    label = train_data['label'].values
    
    # model training
    train_ctr_model(model_name,dataset,model_path,batch_size,epochs)