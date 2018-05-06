# -*- coding: utf-8 -*-
"""
Multi Layer Perceptron Example For TensorFlow
"""

# --------------------------------------------------
# import
# --------------------------------------------------
import math
from sklearn.utils import shuffle
import tensorflow as tf
import common


# --------------------------------------------------
# TensorFlow用関数
# --------------------------------------------------
def he_initial(n_in, n_out):
    """
    戻り値
        Heの初期値
    """
    stddev = math.sqrt(2.0/n_in)
    return tf.random_normal([n_in, n_out], stddev=stddev)


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

    # 入力と教師データの設定
    x = tf.placeholder(tf.float32, shape=[None, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    # ドロップアウトしない確率
    keep_prob = tf.placeholder(tf.float32)

    # 入力層-隠れ層
    # 全結合層
    Wi = tf.Variable(he_initial(n_in, n_hidden_list[0]))
    bi = tf.Variable(tf.zeros([n_hidden_list[0]]))
    z = tf.matmul(x, Wi) + bi
    # ReLU
    z = tf.nn.relu(z)

    # 隠れ層-隠れ層
    for i in range(len(n_hidden_list) - 1):
        n_in_h = n_hidden_list[i]
        n_out_h = n_hidden_list[i+1]
        Wh = tf.Variable(he_initial(n_in_h, n_out_h))
        bh = tf.Variable(tf.zeros([n_out_h]))
        z = tf.matmul(z, Wh) + bh
        z = tf.nn.relu(z)
    z = tf.nn.dropout(z, keep_prob)

    # 隠れ層-出力層
    Wo = tf.Variable(he_initial(n_hidden_list[-1], n_out))
    bo = tf.Variable(tf.zeros([n_out]))
    z = tf.matmul(z, Wo) + bo
    y = tf.nn.softmax(z)


    # --------------------------------------------------
    # 学習
    # --------------------------------------------------
    # Loss関数とOptimizerの設定
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y+1e-10), axis=1))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # 学習
    epochs = common.epochs
    batch_size = common.batch_size
    n_batches = len(X_train) // batch_size

    # セッション初期化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step,
                     feed_dict={x:X_[start:end],
                                t:Y_[start:end],
                                keep_prob:0.5})


    # --------------------------------------------------
    # 検証
    # --------------------------------------------------
    # Accuracyの設定
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss_train = cross_entropy.eval(session=sess, feed_dict={x:X_train, t:Y_train, keep_prob:1})
    loss_test = cross_entropy.eval(session=sess, feed_dict={x:X_test, t:Y_test, keep_prob:1})

    acc_train = accuracy.eval(session=sess, feed_dict={x:X_train, t:Y_train, keep_prob:1})
    acc_test = accuracy.eval(session=sess, feed_dict={x:X_test, t:Y_test, keep_prob:1})


    # --------------------------------------------------
    # 結果のグラフ化
    # --------------------------------------------------
    common.plot_result(loss_train, loss_test, acc_train, acc_test)
    print(acc_test)


if __name__ == "__main__":
    main()
