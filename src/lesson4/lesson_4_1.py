from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist       # データ読み込み用
from tensorflow.keras.utils import to_categorical # データ読み込み用

# Inputレイヤーからスタート（返り値はテンソル）
inputs = Input(shape=(784,))

# レイヤークラスのインスタンスはテンソルを引数に取れる（返り値はテンソル）
# InputレイヤーとDenseレイヤー(1層目)を接続
x = Dense(128, activation='relu')(inputs)

# Denseレイヤー(1層目)とDenseレイヤー(2層目)を接続
x = Dense(64, activation='relu')(x)

# レイヤーのインスタンス化を切り分けることももちろん可能
output_layer = Dense(10, activation='softmax')

# (別のモデル構成時にこのレイヤーを指定・再利用することも可能になる)
# Denseレイヤー(2層目)とDenseレイヤー(3層目)を接続
predictions = output_layer(x)

# Modelクラスを作成（入力テンソルと出力テンソルを指定すればよい）
# これで、「(784,)のInputを持つDense3層」構成のモデルが指定される
model = Model(inputs=inputs, outputs=predictions)

# 以降はSequentialと同じ
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train)

print(model)
