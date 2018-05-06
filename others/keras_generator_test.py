# -*- coding: utf-8 -*-
"""
@author: ytgw
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.utils import Sequence
from keras.optimizers import Adam


def create_dateframe(time_delta_name):
    """
    引数
        時間差の項目名

    戻り値
        DataFrame
    """
    date = "Date"
    date_parser = lambda date: pd.datetime.strptime(date, "%Y-%m-%d")

    df = pd.read_csv("us-dollar-exchange-rate.csv", parse_dates=[date], date_parser=date_parser)
    df.dropna(inplace=True)
    df[time_delta_name] = (df[date] - df[date].values[0]).dt.total_seconds() // (24*60*60)
    df = df.drop(date,axis=1)
    df = (df - df.mean()) / df.std()

    return df


class TimeSeriesGenerator(Sequence):
    def __init__(self, data, target, past, future, step, batch_size, shuffle, use_delta_time=False, delta_time=None):
        """
        引数
            data: input data(np.array)
            target: target data(np.array)
            past: past length(int >= 0)
            future: future length(int >= 0)
            step: sampling step(int >= 1)
            batch_size: batch size(int >= 1)
            shuffle: shuffle or not(bool)
            use_delta_time: use delta time or not(bool)
            delta_time: delta time(np.array)
        """
        self.data = data
        self.target = target
        self.step = step
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_delta_time = use_delta_time
        self.delta_time = delta_time

        self.past = step * past
        self.future = step * future

        # shuffle index
        shuffled_index = np.array(range(len(target)))
        if self.future > 0:
            self.shuffled_index = shuffled_index[self.past : -1-self.future]
        else:
            self.shuffled_index = shuffled_index[self.past :]
        self.shuffle_idx()

        # for self.__len__() method
        n_data = len(self.shuffled_index)
        if n_data % self.batch_size == 0:
            self.n_per_epoch = len(self.shuffled_index) // self.batch_size
        else:
            self.n_per_epoch = len(self.shuffled_index) // self.batch_size + 1

    def __getitem__(self, index):
        """
        引数
            index: batch index(0<= int <=self.__len__())
        戻り値
            input data(np.array)
            target data(np.array)
        """
        # 最終インデックスかどうかでバッチサイズは変更
        # 使用インデックスを取得
        if index < self.n_per_epoch:
            use_indices = self.shuffled_index[index * self.batch_size : (index+1)*self.batch_size]
        else:
            use_indices = self.shuffled_index[index * self.batch_size :]

        # 入力データの箱を確保
        n_time = past + future + 1
        if self.use_delta_time:
            n_features = len(input_names) + 1
        else:
            n_features = len(input_names)
        X_batch = np.zeros(shape=(len(use_indices), n_time, n_features))

        # 使用インデックス毎に入力時系列データを作成
        for i, idx in enumerate(use_indices):
            sidx = idx - self.past
            eidx = idx + self.future + 1
            if self.use_delta_time:
                delta_time = (self.delta_time[sidx:eidx:self.step] - self.delta_time[idx]).reshape(-1,1)
                X_batch[i] =  np.concatenate((delta_time, self.data[sidx:eidx:self.step, :]), axis=1)
            else:
                X_batch[i] = self.data[sidx:eidx:self.step, :]

        # 出力データを作成
        y = np.array(self.target[use_indices])

        return X_batch, y

    def __len__(self):
        """
        戻り値
            number of batch per epoch
        """
        return self.n_per_epoch

    def on_epoch_end(self):
        """
        エポック終了毎にデータをシャッフル

        戻り値なし
        """
        self.shuffle_idx()

    def shuffle_idx(self):
        """
        データをシャッフルする

        戻り値なし
        """
        if self.shuffle:
            self.shuffled_index = shuffle(self.shuffled_index)

    def get_target(self):
        """
        戻り値
            サイズを合わせた出力データ
        """
        if self.future > 0:
            return self.target[self.past : -1-self.future]
        else:
            return self.target[self.past:]

    def sort_idx(self):
        """
        データを時系列順にソートする

        戻り値なし
        """
        self.shuffled_index.sort()

    def set_shuffle_sort(self, shuffle):
        """
        エポックごとのシャッフルの有無を変更し、データをソートorシャッフルする

        戻り値なし
        """
        if shuffle:
            self.shuffle = True
            self.shuffle_idx()
        else:
            self.shuffle = False
            self.shuffled_index.sort()


if __name__ == "__main__":
    # データフレーム作成
    target_name = "Japanese Yen to One U.S. Dollar"
    input_names = ["Brazilian Reals to One U.S. Dollar",
                   "Chinese Yuan to One U.S. Dollar",
                   "Hong Kong Dollars to One U.S. Dollar",
                   "Indian Rupees to One U.S. Dollar",
                   "Singapore Dollars to One U.S. Dollar",
                   "Swedish Kronor to One U.S. Dollar",
                   "Thai Baht to One U.S. Dollar"]
    time_delta_name = "time_delta_day"

    df = create_dateframe(time_delta_name)

    # Keras用のGenerator作成
    past = 0
    future = 0
    step = 1
    n_time = past+future+1
    use_delta_time = True
    if use_delta_time:
        n_feature = len(input_names) + 1
    else:
        n_feature = len(input_names)

    train_gen = TimeSeriesGenerator(data=df[input_names].values,
                                       target=df[target_name].values,
                                       past=past,
                                       future=future,
                                       step=step,
                                       batch_size=512,
                                       shuffle=True,
                                       use_delta_time=use_delta_time,
                                       delta_time=df[time_delta_name].values)

    # DNNモデル定義
    model = Sequential()
    model.add(GRU(input_shape=(n_time, n_feature), units=15))
    model.add(Dense(units=15, activation="relu"))
    model.add(Dense(units=1, activation="linear", use_bias=False))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=1e-2))

    # モデル学習
    epochs = 30
    hist = model.fit_generator(train_gen, epochs=epochs, verbose=1)
    plt.plot(hist.history["loss"], "o-")
    plt.show()

    # 学習結果の確認
    y_true = train_gen.get_target()
    # 予測前にソートしないとシャッフルされたまま結果が来る
    train_gen.set_shuffle_sort(False)
    y_pred = model.predict_generator(train_gen).reshape(-1)
    plt.plot(y_true,"r", label="true")
    plt.plot(y_pred, "b", label="pred", alpha=0.5)
    plt.legend()
    plt.show()

    plt.plot(y_true, y_pred, "o")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.show()

