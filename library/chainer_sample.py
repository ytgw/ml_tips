# -*- coding: utf-8 -*-
"""
Multi Layer Perceptron Example For Chainer
"""

# --------------------------------------------------
# import
# --------------------------------------------------
import numpy as np
import chainer
from chainer import training, Variable, iterators, optimizers, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.initializers import HeNormal
import common


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    """
    Main Function
    """
    # 再現性確保
    common.set_random_state(0)

    # --------------------------------------------------
    # データ取得
    # --------------------------------------------------
    dataset = "mnist"
    X_train, X_test, Y_train, Y_test = common.read_data(data=dataset, one_hot=False)
    n_in = X_train.shape[1]
    if dataset == "mnist":
        n_out = 10

    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.int)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.int)

    train_data = [(X_train[i], Y_train[i]) for i in range(len(X_train))]


    # --------------------------------------------------
    # モデル構築
    # --------------------------------------------------
    n_hidden_list = common.n_hidden_list

    class MLP(ChainList):
        """
        Multi Layer Perceptron
        """
        def __init__(self, n_in, n_hidden_list, n_out):
            """
            引数
                n_in:入力層ユニット数、n_hidden_list:隠れ層ユニット数のリスト、n_out:出力層ユニット数

            戻り値なし
            """
            super(MLP, self).__init__()
            with self.init_scope():
                # 入力層=隠れ層、隠れ層-隠れ層
                for n_hidden in n_hidden_list:
                    self.add_link(L.Linear(None, n_hidden, initialW=HeNormal()))

                # 隠れ層-出力層
                self.add_link(L.Linear(None, n_out, initialW=HeNormal()))

        def __call__(self, x):
            """
            引数
                モデル入力

            戻り値
                モデル出力(出力は全結合層)
            """
            link_list = list(self.children())

            # 入力層=隠れ層
            x = link_list[0](x)
            x = F.relu(x)

            # 隠れ層-隠れ層
            for i,link in enumerate(link_list[1:-1]):
                x = link(x)
                x = F.relu(x)
            x = F.dropout(x, 0.5)

            # 隠れ層-出力層
            x = link_list[-1](x)
            return x

    model = L.Classifier(MLP(n_in, n_hidden_list, n_out))


    # --------------------------------------------------
    # 学習
    # --------------------------------------------------
    train_iter = iterators.SerialIterator(train_data,
                                          batch_size=common.batch_size,
                                          shuffle=True)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (common.epochs,'epoch'), out='chainer_result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'elapsed_time']))

    with chainer.using_config('train', True):
        trainer.run()


    # --------------------------------------------------
    # 検証
    # --------------------------------------------------
    with chainer.using_config('train', False):
        model(Variable(X_train), Variable(Y_train))
        loss_train = model.loss.data
        acc_train = model.accuracy.data

        model(Variable(X_test), Variable(Y_test))
        loss_test = model.loss.data
        acc_test = model.accuracy.data


    # --------------------------------------------------
    # 結果のグラフ化
    # --------------------------------------------------
    common.plot_result(loss_train, loss_test, acc_train, acc_test)
    print(acc_test)


if __name__ == "__main__":
    main()
