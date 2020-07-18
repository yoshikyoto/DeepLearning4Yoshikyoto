from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Permute, Activation, \
    concatenate, dot
import numpy as np

# 下記テキストのプログラム
# https://github.com/matsuolab-edu/dl4us/blob/master/lesson4/lesson4_sec2_exercise.ipynb


def load_data(file_path):
    tokenizer = Tokenizer(filters="")
    whole_texts = []
    for line in open(file_path, encoding='utf-8'):
        # 行の最初に <s> 、最後に </s> をつける
        # これにより、 <s> が文章の最初、 </s> が文章の最後という意味になる
        whole_texts.append("<s> " + line.strip() + " </s>")

    tokenizer.fit_on_texts(whole_texts)

    return tokenizer.texts_to_sequences(whole_texts), tokenizer


encoder_model_filename = 'lesson_4_4_encoder_model.h5'
decoder_model_filename = 'lesson_4_4_decoder_model.h5'
attention_model_filename = 'lesson_4_4_attention_model.h5'

# 読み込み＆Tokenizerによる数値化
x_train, tokenizer_en = load_data('data/train.en')
y_train, tokenizer_ja = load_data('data/train.ja')

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

# データを学習用とテスト用に分ける
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.02, random_state=42)

# パディング
# 文の長さが違うのでゼロパディングして長さを揃える必要がある
x_train = pad_sequences(x_train, padding='post')
y_train = pad_sequences(y_train, padding='post')

seqX_len = len(x_train[0])
seqY_len = len(y_train[0])

# 単語の one-hot表現から変換する embed表現の次元数
emb_dim = 256
hid_dim = 256
att_dim = 256

# 保存されたモデルがあれば読み込む
encoder_model = None
decoder_model = None
attention_model = None

try:
    encoder_model = load_model(encoder_model_filename)
    decoder_model = load_model(decoder_model_filename)
    attention_model = load_model(attention_model_filename)
    print("保存されたモデルを使います")
except IOError:
    print("保存されたモデルが無いので学習から始めます")

# どれか読み込めなかった場合は学習をやりなおす
if encoder_model is None or decoder_model is None or attention_model is None:
    # 符号化器
    # Inputレイヤー（返り値としてテンソルを受け取る）
    encoder_inputs = Input(shape=(seqX_len,))

    # モデルの層構成（手前の層の返り値テンソルを、次の接続したい層に別途引数として与える）
    # InputレイヤーとEmbeddingレイヤーを接続（+Embeddingレイヤーのインスタンス化）
    # shape: (seqX_len,)->(seqX_len, emb_dim)
    # mask_zero=True を指定することで、ゼロパディングした部分を無視するようにしている
    encoder_embedded = Embedding(en_vocab_size, emb_dim, mask_zero=True)(encoder_inputs)

    # EmbeddingレイヤーとLSTMレイヤーを接続（+LSTMレイヤーのインスタンス化）
    # shape: (seqX_len, emb_dim)->(hid_dim, )
    # return_state=True を指定することで、 *encoder_states に隠れ状態を返す
    # `output = LSTM()(x)`
    # `output, state_h, state_c = LSTM(return_state=True)(x)`
    # 今回の場合、 output は使わず、 *encoder_states に state_h, state_c が入る
    encoded_seq, *encoder_states = LSTM(
        hid_dim,
        return_sequences=True,
        return_state=True,
    )(encoder_embedded)

    # 復号化器
    # Inputレイヤー（返り値としてテンソルを受け取る）
    decoder_inputs = Input(shape=(seqY_len,))

    # モデルの層構成（手前の層の返り値テンソルを、次の接続したい層に別途引数として与える）
    # InputレイヤーとEmbeddingレイヤーを接続
    # 後で参照したいので、レイヤー自体を変数化
    decoder_embedding = Embedding(ja_vocab_size, emb_dim)
    # shape: (seqY_len,)->(seqY_len, emb_dim)
    decoder_embedded = decoder_embedding(decoder_inputs)

    # EmbeddingレイヤーとLSTMレイヤーを接続（encoder_statesを初期状態として指定）
    # 後で参照したいので、レイヤー自体を変数化
    decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
    # shape: (seqY_len, emb_dim)->(seqY_len, hid_dim)
    decoded_seq, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)

    # Attentionレイヤー
    score_dense = Dense(hid_dim)
    score = score_dense(decoded_seq)
    score = dot([score, encoded_seq], axes=(2, 2))
    attention = Activation('softmax')(score)
    context = dot([attention, encoded_seq], axes=(2, 1))
    concat = concatenate([context, decoded_seq], axis=2)
    attention_dense = Dense(att_dim, activation='tanh')
    attentional = attention_dense(concat)
    output_dense = Dense(ja_vocab_size, activation='softmax')
    outputs = output_dense(attentional)

    # モデル構築（入力は符号化器＆復号化器、出力は復号化器のみ）
    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    # 今回は、sparse_categorical_crossentropy（正解ラベルとしてone_hot表現のベクトルでなく数値を受け取るcategorical_crossentropy）を使用

    # モデルの学習は、教師データとしては1つ先の単語を示すデータにする: train_target
    train_target = np.hstack((y_train[:, 1:], np.zeros((len(y_train),1), dtype=np.int32)))

    # 学習する
    model.fit(
        [x_train, y_train],
        np.expand_dims(train_target, -1),
        batch_size=128,
        epochs=10,
        verbose=2,
        validation_split=0.2
    )

    # 今度は、学習したモデルを使って、系列を生成する
    # 学習したモデルは次の単語を予測するモデルなので、
    # 文章を生成するためのモデルを作る必要がある

    # サンプリング用（生成用）のモデルを作成

    # 符号化器（学習時と同じ構成、学習したレイヤーを利用）
    encoder_model = Model(encoder_inputs, [encoded_seq] + encoder_states)

    # 復号化器
    # decorder_lstm の初期状態指定用(h_t, c_t)
    decoder_states_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]

    decoder_inputs = Input(shape=(1,))

    # 学習済みEmbeddingレイヤーを利用
    decoder_embedded = decoder_embedding(decoder_inputs)

    # 学習済みLSTMレイヤーを利用
    decoded_seq, *decoder_states = decoder_lstm(
        decoder_embedded,
        initial_state=decoder_states_inputs
    )

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoded_seq] + decoder_states
    )

    # Attentionレイヤー
    encoded_seq_in, decoded_seq_in = Input(shape=(seqX_len, hid_dim)), Input(shape=(1, hid_dim))
    score = score_dense(decoded_seq_in)
    score = dot([score, encoded_seq_in], axes=(2, 2))
    attention = Activation('softmax')(score)
    context = dot([attention, encoded_seq_in], axes=(2, 1))
    concat = concatenate([context, decoded_seq_in], axis=2)
    attentional = attention_dense(concat)
    attention_outputs = output_dense(attentional)
    attention_model = Model([encoded_seq_in, decoded_seq_in], [attention_outputs, attention])

    print("モデルを保存します")
    encoder_model.save(encoder_model_filename)
    decoder_model.save(decoder_model_filename)
    attention_model.save(attention_model_filename)


def decode_sequence(input_seq, bos_eos, max_output_length=1000):
    encoded_seq, *states_value = encoder_model.predict(input_seq)

    target_seq = np.array(bos_eos[0])  # bos_eos[0]="<s>"に対応するインデックス
    output_seq = bos_eos[0][:]
    attention_seq = np.empty((0, len(input_seq[0])))

    while True:
        decoded_seq, *states_value = decoder_model.predict([target_seq] + states_value)
        output_tokens, attention = attention_model.predict([encoded_seq, decoded_seq])
        sampled_token_index = [np.argmax(output_tokens[0, -1, :])]
        output_seq += sampled_token_index
        attention_seq = np.append(attention_seq, attention[0], axis=0)

        if (sampled_token_index == bos_eos[1] or len(output_seq) > max_output_length):
            break

        target_seq = np.array(sampled_token_index)

    return output_seq, attention_seq


detokenizer_en = dict(map(reversed, tokenizer_en.word_index.items()))
detokenizer_ja = dict(map(reversed, tokenizer_ja.word_index.items()))

text_no = 0
input_seq = pad_sequences([x_test[text_no]], seqX_len, padding='post')
bos_eos = tokenizer_ja.texts_to_sequences(["<s>", "</s>"])

output_seq, attention_seq = decode_sequence(input_seq, bos_eos)

print('元の文:', ' '.join([detokenizer_en[i] for i in x_test[text_no]]))
print('生成文:', ' '.join([detokenizer_ja[i] for i in output_seq]))
print('正解文:', ' '.join([detokenizer_ja[i] for i in y_test[text_no]]))
