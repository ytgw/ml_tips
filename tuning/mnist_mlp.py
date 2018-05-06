# -*- coding: utf-8 -*-
"""
@author: ytgw
"""

# --------------------------------------------------
# import
# --------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# --------------------------------------------------
# functions
# --------------------------------------------------
def load_data(use_ratio=None):
    """
    引数
        use_ratio: MNISTデータを使用する割合(0 <= float <= 1)
    戻り値
        (X_train, Y_train), (X_test, Y_test)
    """
    # データ読み込み
    mnist = datasets.fetch_mldata("MNIST original", data_home=".")
    X = mnist.data
    y = mnist.target

    # 出力のOne Hot表現化
    Y = np.eye(10)[y.astype(int)]

    # データを訓練と検証に分割
    n_train = 60000
    n_test = len(X) - n_train
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    if use_ratio is not None:
        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
        X_train = X_train[:int(use_ratio*n_train)]
        Y_train = Y_train[:int(use_ratio*n_train)]

        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
        X_test = X_test[:int(use_ratio*n_test)]
        Y_test = Y_test[:int(use_ratio*n_test)]

    return (X_train, Y_train), (X_test, Y_test)


def define_mnist_model(hidden_list, drop_list, n_in=784, n_out=10):
    """
    戻り値
        model
    """
    model = Sequential()
    model.add(Dense(input_shape=(n_in,), units=hidden_list[0], activation="relu"))
    model.add(Dropout(drop_list[0]))

    for i in range(1, len(hidden_list)):
        model.add(Dense(units=hidden_list[i], activation="relu"))
        model.add(Dropout(drop_list[i]))

    model.add(Dense(units=n_out, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=["accuracy"])

    return model


class MNIST_MLP():
    """
    MLP Class For MNIST
    """
    def __init__(self, hidden_list, drop_list, epochs, batch_size, val_ratio):
        """
        戻り値なし
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.__model = define_mnist_model(hidden_list, drop_list)

    def fit(self, X, Y, early_patience=10):
        """
        戻り値なし
        """
        X, Y = shuffle(X, Y)
        early_stopping = EarlyStopping(patience=early_patience, verbose=1)
        self.__model.fit(X, Y,
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         validation_split=self.val_ratio,
                         callbacks=[early_stopping],
                         verbose=0)

    def evaluate(self, X, Y):
        """
        戻り値
            [loss, acc]
        """
        loss_acc = self.__model.evaluate(X, Y,
                                         batch_size=self.batch_size,
                                         verbose=0)
        return loss_acc

class CalcPerformance():
    def __init__(self, n_trial=3, data_use_ratio=0.1, objection="error_rate", verbose=True, save_fig=True):
        """
        戻り値なし
        """
        self.n_trial = n_trial
        self.data_use_ratio = data_use_ratio
        self.objection = objection
        self.verbose = verbose
        self.save_fig = save_fig
        self.summary_list = []
        if verbose:
            print("n_trial        :", n_trial)
            print("data_use_ratio :", data_use_ratio)
            print("objection      :", objection)

    def __call__(self, hidden_list, drop_list, epochs, batch_size, val_ratio, early_patience):
        """
        戻り値
            self.objectionによって決まるスカラー値
        """
        # 入力の表示
        input_dict = {"hidden_list":hidden_list,
                      "drop_list":drop_list,
                      "epochs":epochs,
                      "batch_size":batch_size,
                      "val_ratio":val_ratio,
                      "early_patience":early_patience}
        if self.verbose:
            self.print_input(input_dict)

        # データ読み込み
        (X_train, Y_train), (X_test, Y_test) = load_data(use_ratio=self.data_use_ratio)

        # 計算実行
        sum_loss_acc = np.zeros(2)
        for i in range(self.n_trial):
            mnist_mlp = MNIST_MLP(hidden_list, drop_list, epochs, batch_size, val_ratio)
            mnist_mlp.fit(X_train, Y_train, early_patience)
            sum_loss_acc += np.array(mnist_mlp.evaluate(X_test, Y_test))

        loss = sum_loss_acc[0] / self.n_trial
        acc = sum_loss_acc[1] / self.n_trial

        # 出力の表示
        output_dict = {"loss":loss,
                       "accuracy":acc,
                       "error_rate":1-acc}
        if self.verbose:
            self.print_output(output_dict)

        # 結果の保存
        self.summary_list.append({"input":input_dict, "output":output_dict})

        # グラフの保存
        if self.save_fig:
            self.plot_output(save_flag=True)

        # 戻り値算出
        return output_dict[self.objection]

    def print_input(self, input_dict):
        """
        戻り値なし
        """
        for key, value in input_dict.items():
            print("{:<15}:".format(key), value)

    def print_output(self, output_dict):
        """
        戻り値なし
        """
        for key, value in output_dict.items():
            print("{} Times Average".format(self.n_trial), end=" ")
            print("{:<10} : {:.3f}".format(key, value))

    def print_best(self):
        """
        戻り値なし
        """
        # 共通条件の表示
        print("n_trial        :", self.n_trial)
        print("data_use_ratio :", self.data_use_ratio)
        print("objection      :", self.objection)
        # 最適化結果の取得
        objection_list = [summary["output"][self.objection] for summary in self.summary_list]
        min_idx = np.argmin(np.array(objection_list))
        # 入力の表示
        input_dict = self.summary_list[min_idx]["input"]
        self.print_input(input_dict)
        # 出力の表示
        output_dict = self.summary_list[min_idx]["output"]
        self.print_output(output_dict)

    def plot_output(self, save_flag):
        """
        戻り値なし
        """
        output_list = [summary["output"] for summary in self.summary_list]
        for key in output_list[0].keys():
            value_list = [output[key] for output in output_list]
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(value_list, "o-")
            ax.set_title(key)

            if save_flag:
                dname = os.path.dirname( os.path.abspath( __file__ ) ) + "/output_graph/"
                fname = dname + "{:0>4}_{}.png".format(len(output_list), key)
                plt.savefig(fname)
            else:
                plt.show()


if __name__ == "__main__":
    min_function = CalcPerformance(n_trial=2, data_use_ratio=0.01, objection="error_rate", verbose=True)
    min_function(hidden_list=[50,50], drop_list=[0.1, 0.1], epochs=3, batch_size=512, val_ratio=0.1, early_patience=1)
    min_function(hidden_list=[50,50], drop_list=[0.5, 0.5], epochs=3, batch_size=512, val_ratio=0.1, early_patience=1)
    min_function.plot_output(save_flag=False)
    print("Best Result")
    min_function.print_best()
