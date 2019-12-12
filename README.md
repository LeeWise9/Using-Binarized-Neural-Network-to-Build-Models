# Using-Binarized-Neural-Network-to-Build-Models
This project will explore how to build a binarized neural network. Take MNIST and traffic signs recognition for example. The code is based on keras and runs on GPU.

本项目将探讨如何使用二值化神经网络（BNN）搭建模型。本项目以mnist数据集和GTSRB（德国交通指示牌）为例。代码基于keras编写，支持GPU加速。

本项目主要包含以下几个部分：<br>
* 0.二值化神经网络简介；<br>
* 1.二值化神经网络计算原理；<br>
* 2.二值化神经网络训练方法；<br>
* 3.二值化神经网络的Keras实现；<br>
* 4.二值化神经网络识别手写数字；<br>
* 5.二值化神经网络识别交通指示牌。<br>


## 0. 二值化神经网络简介<br>
为了将神经网络部署到诸如单片机这种算力有限的设备上，[二值化神经网络](https://arxiv.org/abs/1602.02830)被提出。

一句话概括 BNN 与普通神经网络的区别：训练更困难，部署更简单，计算更高效。

二值网络是将权值 W 和隐藏层激活值二值化为 1 或者 -1。

比如二值化的 2D 卷积核可由下图表示（16 个 3x3 的卷积核）：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/bnn%E6%9D%83%E5%80%BC.png" alt="Sample"  width="250">
</p>

通过二值化操作，模型的参数占用更小的存储空间（内存消耗理论上减少为原来的1/32倍，从float32到1bit）；同时可利用位操作来代替网络中的乘加运算，大大降低了预测过程的运算时间。

需要注意的是，二值化网络得到的模型权重值为二值的，但是训练过程中参与误差计算的梯度值是非二值化的，因为使用浮点数计算才能保证计算和训练的精度。

权重值的改变意味着信息的损失，这也意味着训练二值化的网络普通网络更加困难。但是考虑到训练完成后，模型更小计算速度更快，部署成本更低，多花点时间做训练还是值得的。

下图是在 CIFAR-10 数据集上训练普通神经网络和二值化神经网络的误差下降曲线：（虚线为训练误差，实线为测试误差）<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/bnntrain.png" alt="Sample"  width="500">
</p>

由于二值网络只是将网络的参数和激活值二值化，并没有改变网络的结构。因此关注重点是如何二值化，以及二值化后参数如何更新。


## 1. 二值化神经网络计算原理<br>
二值化网络的计算重点在于梯度计算及梯度传递。

### 浮点数的二值化方法<br>
对任意一个 32 位浮点数 x，其二值化方法为取其符号：x 不小于 0 时取 1，小于 0 时取 -1。
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E4%BA%8C%E5%80%BC%E5%8C%96%E6%96%B9%E6%B3%951.png" alt="Sample"  width="300">
</p>

二值化操作如图所示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E4%BA%8C%E5%80%BC%E5%8C%96%E6%96%B9%E6%B3%952.png" alt="Sample"  width="500">
</p>

### 梯度传递计算方法<br>
虽然BNN 训练方法使用二值化的权值和激活值来计算参数梯度。但梯度不得不用其高精度的实际值，因为随机梯度下降（SGD）计算的梯度值量级很小，而且在累加过程中具有噪声，这种噪声是服从正态分布的，因此这种算子需要保持足够高的精度。此外，在计算梯度的时候给权值和激活值添加噪声具有正则化作用，可以防止过拟合。

符号函数sign 的导数为零，显然进无法行反向传播运算。因此，在反传过程中需要对符号函数进行松弛求解。

假设q 的梯度为：<br>
<p align="center">
	<img src="https://img-blog.csdn.net/20180427221046833" alt="Sample"  width="40">
</p>

其中，C 为损失函数，已知 q 的梯度，那么 r 的梯度，即 C 对 r 的求导公式如下：<br>
<p align="center">
	<img src="https://img-blog.csdn.net/20180427221418986" alt="Sample"  width="150">
</p>

其中 ，1|r|<=1  的计算公式为 Htanh，这也是函数变得可求导的原因，具体如下：<br>
<p align="center">
	<img src="https://img-blog.csdn.net/2018042722191180" alt="Sample"  width="400">
</p>

即当r 的绝对值小于1时，r 的梯度等于 q 的梯度，否则 r 的梯度为 0 。可以用下图表示：<br>
<p align="center">
	<img src="https://img-blog.csdn.net/20180427222533395" alt="Sample"  width="500">
</p>

梯度传递操作如图所示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E4%BA%8C%E5%80%BC%E5%8C%96%E6%96%B9%E6%B3%953.png" alt="Sample"  width="500">
</p>


## 2. 二值化神经网络训练方法<br>
二值化神经网络的训练和普通网络整体步骤相同，包括权值前传、梯度反传和权值更新。

先解释以下各个变量符号：<br>
Wk ：第 k 层的权值；<br>
ak ：第 k 层的激活函数值；<br>
θk ：第 k 层的 BN 参数；<br>
sk ：a(k-1) 和 Wk 的积（一个中间变量）。<br>
以上变量如果前方添加 "g" 则表示计算梯度，如果包含上角标 "b" 则表示计算二值化值。<br>

### 前传<br>
前传的计算步骤如下图所示：<br>
<p align="center">
	<img src="https://img-blog.csdn.net/20180428143438609" alt="Sample"  width="500">
</p>

当计算层 k 由 1 增大为 L 时：<br>
1. 对 Wk 做二值化操作得 Wbk；<br>
2. 将 Wbk 与 ab(k-1) 相乘得 sk；<br>
3. 将 sk 做 BN 得 ak，注意 ak、sk 和 θk 都不是二值的；<br>
4. 对 ak 做二值化得 abk。<br>

### 梯度反传<br>
反传的计算步骤如下图所示：<br>
<p align="center">
	<img src="https://img-blog.csdn.net/20180428144519222" alt="Sample"  width="500">
</p>

当计算层 k 由 L 减小为 1 时：<br>
1. 如果 k<L（即：K=L 时不做此步）：将 gbak 与 Htanh(ak) 相乘得 gak；<br>
2. 对 gak、sk 和 θk 做反 BN 得其梯度值；<br>
3. 将 gsk 和 Wbk 相乘得 gba(k-1)；<br>
4. 将 gsk 的转置与第 ab(k-1) 相乘得 gWbk。<br>

计算出权值 W 的梯度之后，就可以更新了，更新权值是网络训练的最终目标。

### 更新权值<br>
更新权值对计算步骤如图所示：（t 表示更新轮数）<br>
<p align="center">
	<img src="https://img-blog.csdn.net/20180428144806929" alt="Sample"  width="500">
</p>

当计算层 k 由 1 增大为 L 时：<br>
1. 由 θk、η（学习率） 和 gθk 更新 θk；<br>
2. 由 Wk、学习率和 gWbk 更新 Wk，并限定变化范围；<br>
3. 更新学习率；<br>


## 3. 二值化神经网络的Keras实现<br>
将二值化神经网络用keras实现，主要包括全连接神经网络和卷积神经网络。

实现神经网络二值化，需要构建了 BinaryDense、BinaryConv2D，还要注意几个技术细节。这里解释一下二值化函数（binarize）、二值化的激活函数（binary_tanh）、二值化的 Dropout 函数（DropoutNoScale）等。

### 二值化函数<br>
```python
def binarize(W, H=1):
    Wb = H * binary_tanh(W / H)
    return Wb
```

该函数将 [-H, H] 之间的值转换为 -H 或者 H，实现二值化操作。

### 二值化激活函数（binary_tanh）<br>
```python
def round_through(x):
    '''
    输入 x∈[ 0, 1 ]
    rounded 取值为 0 或 1
    不计算(rounded - x)的梯度
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)

def _hard_sigmoid(x):
    '''
    二值化的 sigmoid 函数
    用折线替代曲线
    输出值域为[ 0, 1 ]
    '''
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)

def binary_tanh(x):
    '''
    二值化的 tanh 函数
    输出值域为[ -1, 1 ]
    ''' 
    return 2 * round_through(_hard_sigmoid(x)) - 1
```

有一个很重要的计算技巧这里重点解释一下：

在前向传播(forward propagation)时, 二值化期望的输出如下：<br>
x <= 0.0， y = -1；<br>
x >  0.0， y = 1。

在后向传播(backward propagation)求梯度时, 期望的规则如下：<br>
当 x <= -1，    y = -1；<br>
当 -1 < x < 1， y = x；<br>
当 x > 1，      y = 1。

显然，前向传播的法则和后向传播的期望是不相同的。Keras 和 TensorFlow 会按照前向传播法则如实反向计算梯度值，但是我们期望反向传播时按照新的法则计算。<br>

鉴于以上矛盾，有了 round_through 函数，它的功能是：<br>
前向传播时，返回值 rounded ，即对 x 取整，得到 0 或 1；<br>
反向传播计算梯度时，不计算 (rounded - x) 部分的梯度，而是计算 x 的梯度，避免梯度为0。<br>

### 二值化 Dropout 函数（DropoutNoScale）<br>
```python
class DropoutNoScale(Dropout):
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,training=training)
        return inputs
```

普通神经网络中的 Dropout 的运行机制是，随机使 rate 比率的神经元失活，并对剩下的权值除以（1-rate）。二值化神经网络不需要这个步骤，所以要乘以（1-rate）作为补偿。


## 4. 二值化神经网络识别手写数字<br>
该项目给出了二值化的全连接神经网络和卷积神经网络识别手写数字的demo，整体而言，卷积神经网络模型效果更好。

卷积神经网络包含四个卷积层（32x3x3, 64x3x3, 64x3x3, 128x3x3）和两个全连接层（其中一个为输出层），参数量为148,680。<br>
这是卷积神经网络的误差下降曲线：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/BNN_CNN_model.jpg" alt="Sample"  width="500">
</p>

全连接神经网络包含3个隐藏层，每个隐藏层包含 512 个神经元，参数量为937,000。<br>
这是全连接神经网络的误差下降曲线：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/BNN_MLP_model.jpg" alt="Sample"  width="500">
</p>


## 5. 二值化神经网络识别交通指示牌<br>
交通指示牌是我另外一个项目中的内容，读者可以通过[传送门](https://github.com/LeeWise9/Autopilot_Traffic_Sign_Recognition)访问该项目并获取数据。使用二值化神经网络识别交通指示牌，本项目同样给出了全连接神经网络和卷积神经网络。

卷积神经网络包含四个卷积层（128x3x3, 128x3x3, 256x3x3, 256x3x3）和两个全连接层（其中一个为输出层），参数量为867,884，训练准确率达到了96.66%，测试准确率达到了96.88%。<br>
这是卷积神经网络的误差下降曲线：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/sign_BNN_CNN.jpg" alt="Sample"  width="500">
</p>

全连接神经网络包含3个隐藏层，每个隐藏层包含 2048 个神经元，参数量为10,598,572。<br>
这是全连接神经网络的误差下降曲线：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/sign_MLP_CNN.jpg" alt="Sample"  width="500">
</p>

总体来说，卷积神经网络的性能要优于全连接神经网络，而且模型更小。

