import pandas as pd
import os

OUTPUT_DIR = 'output'
HDF = os.path.join(OUTPUT_DIR, 'feat_mat.h5')

def train(max_date_block_num=32, min_date_block_num=3):
    data = pd.read_hdf(HDF, key='feat')
    X_train = data[(data.date_block_num <= max_date_block_num) & (data.date_block_num >= min_date_block_num)].drop(['item_cnt_month'], axis=1)
    Y_train = data[(data.date_block_num <= max_date_block_num) & (data.date_block_num >= min_date_block_num)]['item_cnt_month']
    Y_train = Y_train.clip(0, 20)

    return X_train, Y_train


def val(max_date_block_num=33):
    data = pd.read_hdf(HDF, key='feat')
    X_valid = data[data.date_block_num == max_date_block_num].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == max_date_block_num]['item_cnt_month']
    Y_valid = Y_valid.clip(0, 20)
    return X_valid, Y_valid


def test():
    data = pd.read_hdf(HDF, key='feat')
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    return X_test