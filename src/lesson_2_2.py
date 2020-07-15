import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot

random_state = 42

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=0.5,
    hspace=0.05,
    wspace=0.05
)

for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap='gray')

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

# 入力画像 28x28x1 (縦の画素数)x(横の画素数)x(チャンネル数)
model.add(Conv2D(
    16,
    kernel_size=(5, 5),
    activation='relu',
    kernel_initializer='he_normal',
    input_shape=(28, 28, 1)
))  # 28x28x1 -> 24x24x16

model.add(MaxPooling2D(pool_size=(2, 2)))  # 24x24x16 -> 12x12x16

model.add(Conv2D(
    64,
    kernel_size=(5, 5),
    activation='relu',
    kernel_initializer='he_normal'
))  # 12x12x16 -> 8x8x64

model.add(MaxPooling2D(pool_size=(2, 2)))  # 8x8x64 -> 4x4x64

model.add(Flatten())  # 4x4x64-> 1024
model.add(Dense(10, activation='softmax'))  # 1024 -> 10

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)

# 作成したモデルの確認
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

early_stopping = EarlyStopping(patience=1, verbose=1)
model.fit(
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=100,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)
