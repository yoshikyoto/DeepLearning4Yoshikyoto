from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt

# x: 手書きの画像（28x28の画像）
# y: 正解のラベル（1〜9）
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_train)

# plotしてみる
fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.set_title(str(y_train[i]))
    ax.imshow(x_train[i], cmap='gray')


# 入力画像を行列(28x28)からベクトル(長さ784)に変換 -- 2次元を1次元に変換
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 名義尺度の値をone-hot表現へ変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train)

# 空のモデルを作成
model = Sequential()

# モデルにレイヤーを追加していく

# 一番最初には input_shape で入力データの次元を与える必要がある
# Dence は全結層を表すレイヤー
# activation パラメータで出力ユニットに適用する活性化関数を指定できる
# use_bias パラメータでバイアスを使用するかどうかを決められる。
# このレイヤーで次元が 784 -> 256 となる
model.add(Dense(units=256, input_shape=(784,)))

# Activation は入力に対して活性化関数を適用したものを出力する
# relu は max(0, x) 、つまり値がマイナスなら0に変換するだけ
# sigmod, relu, tanh, softmax がよく使われる
# 一覧は https://keras.io/ja/activations/
model.add(Activation('relu'))

# ここで次元が 256 -> 100 になる
model.add(Dense(units=100))
model.add(Activation('relu'))

# ここで次元が 100 -> 10 になる
model.add(Dense(units=10))

# softmax 関数は出力が規格化されていて、確率として解釈できるので、
# 最後のレイヤーにに使われる事が多い
# 2クラス分類であれば sigmoid 関すが
model.add(Activation('softmax'))

# モデルの学習方法について指定しておく
# loss は損失関数
#  連続値の時は平均二乗誤差が使われることが多い loss='mean_squared_error'
#  離散値の時は交差エントロピーが使われることが多い
#   2クラス交差エントロピー loss='binary_crossentropy'
#   多クラス交差エントロピー loss='categorical_crossentropy'
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# 最後に fit でデータを入れる
model.fit(
    x_train,
    y_train,
    batch_size=1000,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test)
)

# モデルを評価する
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 新しいデータを予測する場合
# classes = model.predict(x_test, batch_size=128)
# print(classes)

# モデルの可視化
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
print(model_to_dot(model, show_shapes=True))
