import os
import glob
import h5py
import keras
import numpy as np
from Name import *
from PIL import Image
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils.utils import get_random_data
def MmNet(input_shape, output_shape):
    model = Sequential()  # 建立模型
    # 第一层
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第二层
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 第三层
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 全连接层
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # 全连接层
    model.add(Dense(output_shape, activation='softmax'))
    print("-----------模型摘要----------\n")  # 查看模型摘要
    model.summary()

    return model

input_shape = (100, 100, 3) #输入
output_shape = 2    #输出
#创建AlexNet模型
model = MmNet(input_shape, output_shape)
print(model.summary())
