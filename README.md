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
二值化网络的计算重点在于梯度计算，前向传播及误差后向传播。
### 0.浮点数的二值化方法<br>
对任意一个32位浮点数x，其二值化方法为取其符号：
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E4%BA%8C%E5%80%BC%E5%8C%96%E6%96%B9%E6%B3%951.png" alt="Sample"  width="50">
</p>




## 2.二值化神经网络训练算法<br>



## 3.二值化神经网络识别手写数字<br>



## 4.二值化神经网络识别交通指示牌<br>



