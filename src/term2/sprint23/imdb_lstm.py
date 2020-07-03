from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
'''
#Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

**Notes**

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.

'''
'''
#IMDBのセンチメント分類タスクでLSTMモデルを学習する。

実際にはデータセットが小さすぎて、TF-IDF + LogRegのようなシンプルで高速な手法と比較して、LSTMの利点はありません。

**注釈**

- RNNは厄介です。バッチサイズの選択が重要、損失とオプティマイザの選択が重要など 一部の構成では収束しない

- 学習中のLSTMの損失減少パターンはCNN/MLPなどで見られるものとは全く異なる場合があります。
'''

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
# この単語数の後にテキストをカットする (max_featuresの上位の最も一般的な単語の中で)
maxlen = 80
batch_size = 512
#batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
# 20000次元（20000単語？）のIMDBデータセットを128次元に落とし込む
model.add(Embedding(max_features, 128))
# LSTMレイヤ　全次元に対して推論する
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# 二値分類なのでDense+sigmoid
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
# 異なるオプティマイザと異なるオプティマイザ設定を使用してみてください。
# -> とりあえずこのまま使う
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          #epochs=15,
          verbose=1, # 経過出力
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
