# -*- coding: utf-8 -*-
"""
hyperopt sample
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
from hyperopt import hp, tpe, fmin
from mnist_mlp import CalcPerformance


if __name__ == "__main__":
    # 最小化関数の設定
    min_function = CalcPerformance(n_trial=2, data_use_ratio=0.1, objection="error_rate", verbose=True, save_fig=False)
    def calc_object_value(args):
        """
        評価関数

        引数
            各種設定値(辞書型)

        戻り値
            min_funtionの戻り値
        """
        print("{}th calculation starts.".format(1+len(min_function.summary_list)))

        hidden_list = [int(args["l1_out"]), int(args["l2_out"])]
        drop_list = [args["l1_drop"], args["l2_drop"]]
        epochs = int(args["epochs"])
        val_ratio = 0.1
        batch_size = 256
        early_patience = 10

        min_val = min_function(hidden_list, drop_list, epochs, batch_size, val_ratio, early_patience)
        print("minimize value : {:.2f}\n".format(min_val))
        return min_val

    # 探索変数の範囲設定
    tune_parameter = {
            "l1_out"  : hp.quniform('l1_out', 5, 100, 1),
            "l2_out"  : hp.quniform('l2_out', 5, 100, 1),
            "l1_drop" : hp.uniform('l1_drop', 0, 0.5),
            "l2_drop" : hp.uniform('l2_drop', 0, 0.5),
            "epochs"  : hp.quniform('epochs', 1, 10, 1),
            }

    # 最適解探索
    print("-------------------- Search Start --------------------")
    best = fmin(calc_object_value,
                tune_parameter,
                algo=tpe.suggest,
                max_evals=15)

    # 結果表示
    print("-------------------- Search End --------------------")
    min_function.plot_output(save_flag=False)
    min_function.print_best()

