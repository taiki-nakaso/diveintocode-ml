'''
[元ファイル](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py)

#Keras (文字レベル) でのシーケンスからシーケンスへの例。

このスクリプトは、基本的な文字レベルのシーケンスからシーケンスへのモデルを実装する方法を示しています。
このスクリプトを、短い英語の文をフランス語の短い文に一文字ずつ翻訳するのに適用します。
このドメインでは単語レベルのモデルが一般的なので、文字レベルの機械翻訳はかなり珍しいことに注意してください。

**アルゴリズムの概要**

- あるドメインからの入力配列（例：英語の文章）と、別のドメインからの対応するターゲット配列（例：フランス語の文章）から開始します。
- エンコーダLSTMは入力配列を2つの状態ベクトルに変換します（LSTMの最後の状態を保持し、出力は破棄します）。
- デコーダLSTMは、ターゲットシーケンスを同じシーケンスに変換するように訓練されますが、将来的には1つのタイムステップ分オフセットされます。
　(「教師強制」と呼ばれるプロセス)
    これは、初期状態としてエンコーダからの状態ベクトルを使用します。
    実質的には、デコーダは入力シーケンスに`targets[t...]`を条件として与えられて`targets[t+1...]`を生成することを学習する。
- 推論モードでは、未知の入力配列を復号化したい場合には、
    - 入力シーケンスを状態ベクトルにエンコードする。
    - サイズ1のターゲットシーケンスから開始します（シーケンスの開始文字のみ）。
    - 状態ベクトルと1文字のターゲットシーケンスをデコーダに送り、次の文字の予測値を生成します。
    - これらの予測値を用いて次の文字をサンプリングします（ここでは単に argmax を使用します）。
    - サンプリングした文字をターゲットシーケンスに追加します。
    - 連続終了文字を生成するか、文字数制限に達するまで繰り返します。

**データのダウンロード**

[英語からフランス語への文のペア。 ] (http://www.manythings.org/anki/fra-eng.zip)

[綺麗な文章ペアのデータセットがたくさんあります。 文章対データセットがたくさんある](http://www.manythings.org/anki/)

**参考文献**

- [ニューラルネットワークによるシーケンス学習](https://arxiv.org/abs/1409.3215)
- [統計的機械翻訳のためのRNNエンコーダデコーダを用いたフレーズ表現の学習](https://arxiv.org/abs/1406.1078)
'''
from __future__ import print_function
'''
futureモジュール:
python3系のコードをpython2系でも動くようにするモジュール。
print_functionは3系の*print関数*を2系の*print文*に対応させる。
'''

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # トレーニング時のバッチサイズ
epochs = 100  # トレーニング時のエポック数
latent_dim = 256  # 入力をベクトルエンコーディングする時の次元数
num_samples = 10000  # トレーニング時のサンプル数（1エポックあたり？train+valid）
# 学習データのパス
data_path = 'fra-eng/fra.txt'

# 入力データのベクトルエンコーディング
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
# 学習データの読み込み
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    # 開始文字: タブ(\t)
    # 終端文字: 改行(\n)
    target_text = '\t' + target_text + '\n'
    # 文章リストに追加
    input_texts.append(input_text)
    target_texts.append(target_text)
    # 文字セットに追加
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

# ソートかける
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
# トークン数　今回は文字レベルでの翻訳なのでトークン数=文字の種類数
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
# 最大テキスト長　paddingの際に利用？
## エンコーダ入力データ
max_encoder_seq_length = max([len(txt) for txt in input_texts])
## デコーダ入力データ/正解データ
max_decoder_seq_length = max([len(txt) for txt in target_texts])
# 出力
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
# 文字と数字の対応を表すdictionary作成
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# 入力・出力データのプレースホルダ
## エンコーダ入力データ: shape(エンコーダ入力サンプル数, エンコーダ入力最大シーケンス長, エンコーダトークン数)
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
## デコーダ入力データ: shape(エンコーダ入力サンプル数, デコーダ入力最大シーケンス長, デコーダトークン数)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
## デコーダ正解データ: shape(エンコーダ入力サンプル数, デコーダ入力最大シーケンス長, デコーダトークン数)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# プレースホルダに突っ込む
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # エンコーダ入力データ処理
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1. # 空白でパディング
    # デコーダ入力データ/正解データ処理
    for t, char in enumerate(target_text):
        # 正解データはデコーダ入力データと1時刻ずれている(1時刻前？)
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # 正解データは開始文字が存在しない
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1. # 空白でパディング
    decoder_target_data[i, t:, target_token_index[' ']] = 1. # 空白でパディング
# 入力シーケンスの定義・処理
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# エンコーダの出力を破棄し状態のみを保持する
encoder_states = [state_h, state_c]

# エンコーダの状態を初期値としてデコーダを構成する
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# デコーダは完全な出力シーケンスを返し、内部状態も返すように設定しています。
# 学習モデルでは戻り値は使用しませんが、推論では使用します。
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# モデルの定義
# エンコーダ入力データ、デコーダ入力データをデコーダ正解データと比較する
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

'''
訓練実行
最適化手法: RMSProp
損失関数: categorical_crossentropy
'''
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data,
           decoder_input_data], # 入力データ
          decoder_target_data, # 正解データ
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# モデルの保存
model.save('s2s.h5')

# 推論モード（サンプリングして実行してみる）
# 手順:
# 1) 入力をエンコードしてデコーダの初期状態を取得する
# 2) この状態で開始文字を入力として1時刻先の出力を得る
#    この出力が次の入力となる
# 3) 新しい入力と状態を用いて予測を繰り返す

# 推論モデルの定義
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 出力シーケンス(number)を読める文(character)にするための逆引きインデックス
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # 入力をベクトルエンコーディングする
    states_value = encoder_model.predict(input_seq)

    # 長さ1のデコーダ入力シーケンスのプレースホルダを定義
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # デコーダ入力シーケンスに初期値として開始文字を入力
    target_seq[0, 0, target_token_index['\t']] = 1.

    # シーケンスのバッチのサンプリングループ（？）
    # (簡単のためバッチサイズは1とする)
    ## 終了フラグ
    stop_condition = False
    ## 出力文
    decoded_sentence = ''
    while not stop_condition:
        # デコーダ入力シーケンスと状態(h, c)から出力を得る
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # characterをサンプリングする
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 終了条件:
        # 文章の長さが学習データの最大長を超えそうになる
        # 終端文字を受け取る
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # デコーダ入力シーケンスの更新 (長さ1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 入力状態値の更新
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # 訓練データからデータを抜き出して翻訳してみる
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
