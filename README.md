# Using-Binarized-Neural-Network-to-Slim-Models
This project will explore how the binary neural network can reduce the computation and the size of the model. Take MNIST and traffic signs recognition for example. The code is based on keras and runs on GPU.

本项目将探讨如何使用二值化神经网络优化模型，减少计算量并减少模型存储空间。本项目以mnist数据集和GTSRB（德国交通指示牌）为例。代码基于keras编写，支持GPU加速。

本项目主要包含四个部分：<br>
* 0.二值化神经网络简介；<br>
* 1.二值化神经网络计算原理；<br>
* 2.二值化神经网络训练算法；<br>
* 3.二值化神经网络识别手写数字；<br>
* 4.二值化神经网络识别交通指示牌。<br>


## 0.二值化神经网络简介<br>
为了将神经网络部署到诸如单片机这种算力有限的设备上，[二值化神经网络](https://arxiv.org/abs/1602.02830)被提出。二值网络是将权值W和隐藏层激活值二值化为1或者-1。通过二值化操作，模型的参数占用更小的存储空间（内存消耗理论上减少为原来的1/32倍，从float32到1bit）；同时利用位操作来代替网络中的乘加运算，大大降低了运算时间。由于二值网络只是将网络的参数和激活值二值化，并没有改变网络的结构。因此关注重点是如何二值化，以及二值化后参数如何更新。同时关注一下如何利用二进制位操作实现GPU加速计算的。


## 1.二值化神经网络计算原理<br>
二值化网络的计算重点在于梯度计算及梯度传递。

### 0.浮点数的二值化方法<br>
对任意一个32位浮点数x，其二值化方法为取其符号：x不小于0时取1，小于0时取-1。
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E4%BA%8C%E5%80%BC%E5%8C%96%E6%96%B9%E6%B3%951.png" alt="Sample"  width="300">
</p>

### 1.梯度计算方法<br>
虽然BNN 训练方法使用二值化的权值和激活值来计算参数梯度。但梯度不得不用其高精度的实际值，因为随机梯度下降（SGD）计算的梯度值量级很小，而且在累加过程中具有噪声，这种噪声是服从正态分布的，因此这种算子需要保持足够高的精度。此外，在计算梯度的时候给权值和激活值添加噪声具有正则化作用，可以防止过拟合。

符号函数sign 的导数为零，显然进无法行反向传播运算。因此，在反传过程中需要对符号函数进行松弛求解。

假设q 的梯度为：
<p align="center">
	<img src="https://img-blog.csdn.net/20180427221046833" alt="Sample"  width="50">
</p>

其中，C 为损失函数，已知q 的梯度，那么r 的梯度，即C 对r 的求导公式如下：
<p align="center">
	<img src="https://img-blog.csdn.net/20180427221418986" alt="Sample"  width="300">
</p>

其中 ，1|r|<=1  的计算公式为Htanh，这也是函数变得可求导的原因，具体如下：
<p align="center">
	<img src="https://img-blog.csdn.net/2018042722191180" alt="Sample"  width="400">
</p>

即当r 的绝对值小于1时，r 的梯度等于q 的梯度，否则r 的梯度为0。可以用下图表示：
<p align="center">
	<img src="https://img-blog.csdn.net/20180427222533395" alt="Sample"  width="500">
</p>




## 2.二值化神经网络训练算法<br>



## 3.二值化神经网络识别手写数字<br>



## 4.二值化神经网络识别交通指示牌<br>



