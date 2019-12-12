# -*- coding: utf-8 -*-
import keras.backend as K


def round_through(x):
    '''
    对 x 中的值取整, 同时使得求梯度的得到的值与原始值的梯度一样
    小技巧, 来自 [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    '''
    当 x <= -1,    y = 0;
    当 -1 < x < 1, y = 0.5 * x + 0.5;
    当 x > 1;      y = 1;
    '''
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)


def binary_tanh(x):
    '''
    在前向传播(forward propagation)时, 输出如下:
        当 x <=  0.0, y = -1
        当 x >  0.0,  y = 1
    
    在后向传播(backward propagation)求梯度时, 求梯度的规则如下:
        2 * _hard_sigmoid(x) - 1 

        当 x <= -1,    y = -1;
        当 -1 < x < 1, y = x;
        当 x > 1;      y = 1;

        当|x| > 1 时, 梯度为 0
    '''
    return 2 * round_through(_hard_sigmoid(x)) - 1


def binarize(W, H=1):
    '''
    二值化操作
    将 [-H, H] 之间的值转换为 -H 或者 H
    '''
    Wb = H * binary_tanh(W / H)
    return Wb