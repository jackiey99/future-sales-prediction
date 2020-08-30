import pandas as pd
import os

OUTPUT_DIR = 'output'
HDF = os.path.join(OUTPUT_DIR, 'feat_mat.h5')
data = pd.read_hdf(HDF, key='feat')


def train(max_date_block_num=32):
    X_train = data[data.date_block_num <= max_date_block_num].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num <= max_date_block_num]['item_cnt_month']
    Y_train = Y_train.clip(0, 20)

    return X_train, Y_train


def val(max_date_block_num=33):
    X_valid = data[data.date_block_num == max_date_block_num].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == max_date_block_num]['item_cnt_month']
    Y_valid = Y_valid.clip(0, 20)
    return X_valid, Y_valid


def test():
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    return X_test