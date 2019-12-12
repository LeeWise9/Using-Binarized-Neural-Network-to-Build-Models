# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:58:24 2019

@author: lenovo
"""

import numpy as np
np.random.seed(37)
import cv2
import pickle
import help_func as h
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D
from sklearn.model_selection import train_test_split

def binary_tanh(x):
    return binary_tanh_op(x)

# 预处理图像
def preprocess_features(X):
    # RGB彩色图-->YUV亮度[:,:,0]
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img,cv2.COLOR_RGB2YUV)[:,:,0],2)for rgb_img in X])
    #直方图均衡化
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)),2)for img in X])
    X = np.float32(X)
    mean_img = np.mean(X, axis=0)
    std_img = (np.std(X, axis=0) + np.finfo('float32' ).eps)  # 为了使除法分母不为0，添加一个极小值
    X -= mean_img   # 图片去均值
    X /= std_img    # 标准化
    return X

H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 128
epochs = 10
channels = 1
img_rows = 32
img_cols = 32 
kernel_size = (3, 3)
pool_size = (2, 2)
use_bias = False

# 学习率变化安排
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# Batch Normalization 参数
epsilon = 1e-6
momentum = 0.9

# 加载交通指示牌数据集, 划分为训练和测试数据
X, y = h.load_traffic_sign_data('./traffic-signs-data/train.p')
X = preprocess_features(X)
X = np.reshape(X,(np.shape(X)[0],1,32,32))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train = X_train.reshape(31367, 1, 32, 32)
#X_test = X_test.reshape(7842, 1, 32, 32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 将类别标签转为 -1 或者 1
classes = np.unique(y_train).shape[0]
Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, classes) * 2 - 1


def build_model(kernel_size,channels,img_rows, img_cols, H, kernel_lr_multiplier, use_bias, epsilon, momentum):
    model = Sequential()
    # conv1
    model.add(BinaryConv2D(128, kernel_size=kernel_size, input_shape=(channels, img_rows, img_cols),
                           data_format='channels_first',
                           H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                           padding='same', use_bias=use_bias, name='conv1'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool1', data_format='channels_first'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
    model.add(Activation(binary_tanh, name='act1'))
    # conv2
    model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                           data_format='channels_first',
                           padding='same', use_bias=use_bias, name='conv2'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool2', data_format='channels_first'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
    model.add(Activation(binary_tanh, name='act2'))
    # conv3
    model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                           data_format='channels_first',
                           padding='same', use_bias=use_bias, name='conv3'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool3', data_format='channels_first'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
    model.add(Activation(binary_tanh, name='act3'))
    # conv4
    model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                           data_format='channels_first',
                           padding='same', use_bias=use_bias, name='conv4'))
    model.add(MaxPooling2D(pool_size=pool_size, name='pool4', data_format='channels_first'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
    model.add(Activation(binary_tanh, name='act4'))
    model.add(Flatten())
    # dense1
    model.add(BinaryDense(256, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
    model.add(Activation(binary_tanh, name='act5'))
    # dense2
    model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))
    
    opt = Adam(lr=lr_start) 
    model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
    model.summary()
    return model

def get_image_generator():
    # 图像增强
    image_datagen = ImageDataGenerator(rotation_range=15.,      # 旋转±15°
                                       width_shift_range=0.1,   # 横向平移
                                       height_shift_range=0.1,  # 纵向平移
                                       zoom_range=0.2)          # 缩放
    return image_datagen

def training_setting(model, image_datagen, x_train, y_train, x_validation, y_validation, epochs, batch_size):
    filepath = "./Models/CNN_weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
    callbacks_list = [checkpoint,lr_scheduler]    # 设置回调函数函数--检查点
    image_datagen.fit(x_train)
    history = model.fit_generator(image_datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=5000, epochs=epochs,
                                  validation_data=(x_validation, y_validation),
                                  callbacks=callbacks_list,verbose=1)
    print(history.history.keys())
    plt.plot(history.history['acc'],label='acc')
    plt.plot(history.history['val_acc'],label='val_acc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('sign_BNN_CNN.jpg',dpi=600)
    plt.show()
    
    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    with open('/trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)     # 保存训练结果（字典）到文件
    return history


if __name__ == "__main__":
    model = build_model(kernel_size,channels,img_rows, img_cols, H, kernel_lr_multiplier, use_bias, epsilon, momentum)
    image_generator = get_image_generator()
    training_setting(model, image_generator, X_train, Y_train, X_test, Y_test, epochs, batch_size)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', score[1])