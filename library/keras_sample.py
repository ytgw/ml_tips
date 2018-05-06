# -*- coding: utf-8 -*-
"""
Multi Layer Perceptron Example For Keras
"""

# --------------------------------------------------
# import
# --------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import print_summary
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
    X_train, X_test, Y_train, Y_test = common.read_data(data="mnist", one_hot=True)
    n_in = X_train.shape[1]
    n_out = Y_train.shape[1]


    # --------------------------------------------------
    # モデル構築
    # --------------------------------------------------
    n_hidden_list = common.n_hidden_list
    model = Sequential()

    # 入力層-隠れ層
    # Denseは全結合層
    model.add(Dense(input_dim=n_in, units=n_hidden_list[0]))
    model.add(Activation(activation="relu"))

    # 隠れ層-隠れ層
    for n_hidden in n_hidden_list[1:]:
        model.add(Dense(units=n_hidden))
        model.add(Activation(activation="relu"))
    model.add(Dropout(rate=0.5))

    # 隠れ層-出力層
    model.add(Dense(units=n_out))
    model.add(Activation("softmax"))

    print_summary(model)


    # --------------------------------------------------
    # 学習
    # --------------------------------------------------
    # Loss関数とOptimizerの設定
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # 学習
    model.fit(X_train, Y_train,
              epochs=common.epochs,
              batch_size=common.batch_size,
              verbose=1)


    # --------------------------------------------------
    # 検証
    # --------------------------------------------------
    loss_train, acc_train = model.evaluate(X_train, Y_train, verbose=0)
    loss_test, acc_test = model.evaluate(X_test, Y_test, verbose=0)


    # --------------------------------------------------
    # 結果のグラフ化
    # --------------------------------------------------
    common.plot_result(loss_train, loss_test, acc_train, acc_test)
    print(acc_test)


if __name__ == "__main__":
    main()
