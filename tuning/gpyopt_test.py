# -*- coding: utf-8 -*-
"""
@author: ytgw
"""
# --------------------------------------------------
# Initial Setting
# --------------------------------------------------
import numpy as np
import tensorflow as tf
import random as rn
import os
# GPUの無効化
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 再現性確保の設定
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# --------------------------------------------------
# import
# --------------------------------------------------
import GPyOpt
from mnist_mlp import CalcPerformance


def x_to_input(x):
    if x.ndim == 2:
        x = x[0]

    hidden_list = [int(x[0]), int(x[1])]
    drop_list = [float(x[2]), float(x[3])]
    epochs = int(x[4])
    val_ratio = 0.1
    batch_size = 256
    early_patience = 10

    return hidden_list, drop_list, epochs, batch_size, val_ratio, early_patience


if __name__ == "__main__":
    # 最小化関数の設定
    min_function = CalcPerformance(n_trial=2, data_use_ratio=0.1, objection="error_rate", verbose=True, save_fig=False)
    def calc_object_value(x):
        """
        評価関数

        引数
            各種設定値(辞書型)

        戻り値
            min_funtionの戻り値
        """
        print("{}th calculation starts.".format(1+len(min_function.summary_list)))
        hidden_list, drop_list, epochs, batch_size, val_ratio, early_patience = x_to_input(x)
        min_val = min_function(hidden_list, drop_list, epochs, batch_size, val_ratio, early_patience)
        print("minimize value : {:.2f}\n".format(min_val))
        return min_val

    # 探索変数の範囲設定
    bounds = [
            {'name': 'l1_out',  'type': 'continuous', 'domain': (5, 100)},
            {'name': 'l2_out',  'type': 'continuous', 'domain': (5, 100)},
            {'name': 'l1_drop', 'type': 'continuous', 'domain': (0, 0.5)},
            {'name': 'l2_drop', 'type': 'continuous', 'domain': (0, 0.5)},
            {'name': 'epochs',  'type': 'continuous', 'domain': (1, 10)}
            ]

    # 事前探索
    print("-------------------- Pre Search Start --------------------")
    opt_mnist = GPyOpt.methods.BayesianOptimization(f=calc_object_value, domain=bounds, initial_design_numdata=5)

    # 最適解探索
    print("-------------------- Search Start --------------------")
    opt_mnist.run_optimization(max_iter=15)

    # 結果表示
    print("-------------------- Search End --------------------")
    min_function.plot_output(save_flag=False)
    min_function.print_best()
