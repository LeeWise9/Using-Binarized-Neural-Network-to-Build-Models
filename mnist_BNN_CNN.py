# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import numpy as np
np.random.seed(37)
import keras.backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D, Clip

def binary_tanh(x):
    return binary_tanh_op(x)

H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 256
epochs = 20
channels = 1
img_rows = 28 
img_cols = 28 
kernel_size = (3, 3)
pool_size = (2, 2)
classes = 10
use_bias = False

# 学习率变化安排
lr_start = 1e-2
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# Batch Normalization 参数
epsilon = 1e-6
momentum = 0.9

# 下载 MNIST 数据集, 分为训练和测试数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 将类别标签转为 -1 或者 1
Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, classes) * 2 - 1


model = Sequential()
# conv1
model.add(BinaryConv2D(32, kernel_size=kernel_size, input_shape=(channels, img_rows, img_cols),
                       data_format='channels_first',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv1'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool1', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
model.add(Activation(binary_tanh, name='act1'))
# conv2
model.add(BinaryConv2D(64, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv2'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool2', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
model.add(Activation(binary_tanh, name='act2'))
# conv3
model.add(BinaryConv2D(64, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv3'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool3', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
model.add(Activation(binary_tanh, name='act3'))
# conv4
model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv4'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool4', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
model.add(Activation(binary_tanh, name='act4'))
model.add(Flatten())
# dense1
model.add(BinaryDense(128, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
model.add(Activation(binary_tanh, name='act5'))
# dense2
model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])

history = history.history
accuracy     = history['acc']
val_accuracy = history['val_acc']
epochs = range(len(accuracy))
plt.plot(epochs,accuracy,label='Training Accuracy')
plt.plot(epochs,val_accuracy,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
#plt.show()
plt.savefig('BNN_CNN_model.jpg',dpi=600)


# 保存和重新加载模型
model.save('cnn.h5')
model = load_model('cnn.h5', custom_objects={'BinaryDense': BinaryDense,
                                             'BinaryConv2D':BinaryConv2D,
                                             'Clip': Clip, 
                                             'binary_tanh': binary_tanh})

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
