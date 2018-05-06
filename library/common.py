# -*- coding: utf-8 -*-
"""
Common Constants And Functions For Multi Layer Perceptron
"""

# --------------------------------------------------
# import
# --------------------------------------------------
import os
import numpy as np
import random as rn
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import backend as K


# --------------------------------------------------
# 定数
# --------------------------------------------------
n_in = 28*28
n_hidden_list = [100, 50, 25]
n_out = 10
epochs = 10
batch_size = 128
n_data = 5000


# --------------------------------------------------
# 関数
# --------------------------------------------------
def set_random_state(seed):
    """
    引数
        乱数シード
    戻り値
        None
    この設定だけでなくコンソールのリセットが必要
    参考　https://faroit.github.io/keras-docs/2.1.5/getting-started/faq/
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    return None


def read_data(data, one_hot):
    """
    引数
        data:データの種類、one_hot:one hot処理の有無
    戻り値
        X_train, X_test, Y_train, Y_test
    """
    if data == "mnist":
        X_train, X_test, Y_train, Y_test = read_mnist_data(one_hot)
    elif data == "can":
        X_train, X_test, Y_train, Y_test = read_can_data(one_hot)

    return X_train, X_test, Y_train, Y_test


def read_mnist_data(one_hot):
    """
    戻り値
        X_train, X_test, Y_train, Y_test
    """
    # データ読み込み
    mnist = datasets.fetch_mldata("MNIST original", data_home=".")
    X = mnist.data
    y = mnist.target

    # データが多すぎるので一部だけ利用
    X, y = shuffle(X, y, random_state=0)
    X = X[:n_data]
    y = y[:n_data]

    # 出力のOne Hot表現化
    if one_hot:
        Y = np.eye(10)[y.astype(int)]
    else:
        Y = y

    # データを訓練と検証に分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

    return X_train, X_test, Y_train, Y_test


def read_can_data(one_hot):
    """
    戻り値
        X_train, X_test, Y_train, Y_test
    """
    return None


def plot_result(loss_train, loss_test, acc_train, acc_test):
    """
    戻り値
        None
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    axL.bar(["train", "test"], [loss_train, loss_test])
    loss_max = max([loss_train, loss_test]) + 0.1
    axL.set_ylim([0,max(1,loss_max)])
    axL.title.set_text("Loss")

    axR.bar(["train", "test"], [acc_train, acc_test])
    axR.set_ylim([0,1])
    axR.title.set_text("Accuracy")
    plt.show()

    return None
