# -*- coding: utf-8 -*-

# 训练二值化的全连接神经网络, 在 MNIST 数据集上面实验.

import numpy as np
np.random.seed(37)

import keras.backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

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


batch_size = 100
epochs = 50
nb_classes = 10

H = 'Glorot'
kernel_lr_multiplier = 'Glorot'

# 网络结构
num_unit = 512
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
drop_in = 0.1
drop_hidden = 0.25

# 下载 MNIST 数据集, 分为训练和测试数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 将类别标签转为 -1 或者 1
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1

model = Sequential()
model.add(DropoutNoScale(drop_in, input_shape=(784,), name='drop0'))
for i in range(num_hidden):
    model.add(BinaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
              name='dense{}'.format(i+1)))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
    model.add(Activation(binary_tanh, name='act{}'.format(i+1)))
    model.add(DropoutNoScale(drop_hidden, name='drop{}'.format(i+1)))
model.add(BinaryDense(10, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
          name='dense'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))

model.summary()

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])

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
plt.savefig('BNN_MLP_model.jpg',dpi=600)

# 保存和重新加载模型
model.save('mlp.h5')

model = load_model('mlp.h5', custom_objects={'DropoutNoScale': DropoutNoScale,
                                             'BinaryDense': BinaryDense,
                                             'Clip': Clip, 
                                             'binary_tanh': binary_tanh})

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
