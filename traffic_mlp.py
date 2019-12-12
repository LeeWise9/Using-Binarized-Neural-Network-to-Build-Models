# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:55:01 2019

@author: Leo
"""


import numpy as np
np.random.seed(37)  # 为了重现
import cv2
import pickle
import help_func as h
import keras.backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, Clip


class DropoutNoScale(Dropout):
    '''Keras 的 Dropout 层会将输入按照 dropout率 进行缩放, 再二值化里面不能进行缩放'''
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

def binary_tanh(x):
    return binary_tanh_op(x)
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

batch_size = 256
epochs = 50

H = 'Glorot'
kernel_lr_multiplier = 'Glorot'

# 网络结构
num_unit = 2048
num_hidden = 3
use_bias = False

# 学习率
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BatchNormalization参数
epsilon = 1e-6
momentum = 0.9

# dropout 参数
drop_in = 0.2
drop_hidden = 0.5

# 加载交通指示牌数据集, 划分为训练和测试数据
X, y = h.load_traffic_sign_data('./traffic-signs-data/train.p')
X = preprocess_features(X)
X = np.reshape(X,(np.shape(X)[0],1024))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 将类别标签转为 -1 或者 1
classes = np.unique(y_train).shape[0]
Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, classes) * 2 - 1


def build_model(classes, drop_in, drop_hidden, num_hidden, num_unit, use_bias, H, kernel_lr_multiplier, epsilon, momentum, lr_start):
    model = Sequential()
    model.add(DropoutNoScale(drop_in, input_shape=(1024,), name='drop0'))
    for i in range(num_hidden):
        model.add(BinaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
                  name='dense{}'.format(i+1)))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
        model.add(Activation(binary_tanh, name='act{}'.format(i+1)))
        model.add(DropoutNoScale(drop_hidden, name='drop{}'.format(i+1)))
    model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
              name='dense'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))
    
    model.summary()
    
    opt = Adam(lr=lr_start)
    model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
    return model


if __name__ == "__main__":
    model = build_model(classes, drop_in, drop_hidden, num_hidden, num_unit, use_bias, 
                        H, kernel_lr_multiplier, epsilon, momentum, lr_start)
    filepath = "./Models/MLP_weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_test, Y_test),
                        callbacks=[lr_scheduler, checkpoint])
    plt.plot(history.history['acc'],label='acc')
    plt.plot(history.history['val_acc'],label='val_acc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('sign_BNN_MLP.jpg',dpi=600)
    plt.show()
    
    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', score[1])
