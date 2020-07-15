import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

# サンプル画像 (5x5)
sample_image = np.array([[1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 0],
                         [0, 1, 1, 0, 0]]
                        ).astype('float32').reshape(1, 5, 5, 1)

# フィルタ
W = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]]).astype('float32').reshape(3, 3, 1, 1)

# 空のモデルを作成
model = Sequential()

model.add(Conv2D(
    1, # フィルターの数
    kernel_size=(3, 3), # フィルターの大きさ
    strides=(1, 1), # フィルターを動かす幅
    padding='valid', # パディング Valid はパディング市内
    input_shape=(5, 5, 1), # 入力のサイズ
    use_bias=False # バイアスは使わない
))
model.layers[0].set_weights([W])

model.predict(sample_image).reshape(3, 3)

