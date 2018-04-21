---
typora-root-url: ./
---

# Tensorflow_youtube

[Luyuan Shi](https://www.youtube.com/channel/UC2Sj-_Y1_F17BwwNJ0wg4EA)

### 第一课 Tensorflow的简介 

`tensorboard` 作为Tensorflow的调试器，非常强大，还可以显示3D图．

`jupyter notebook` 可以作为一个编辑器，用起来也是很流畅的．

> 但是我使用conda虚拟环境诶，这个需要额外配置．暂时放弃．

> 后面发现 `jupyter notebook`的自动补全功能很好用，舍不得放弃．下面是配置方法：
>
> 1. 进入coda环境：`source activate 环境名称 `
> 2. 安装`ipykernel`:`conda install ipykernel  `
> 3. 将环境写入notebook的kernel中: `python -m ipykernel install --user --name 环境名称 --display-name "Python (环境名称)" `
> 4. 打开`notebook`:`jupyter notebook  `
>
> [参考链接](https://blog.csdn.net/u011606714/article/details/77741324)
>
> Tip: 每次新建一个Notebook的时候要结束运行上一个Notebook, 不然会出现奇怪的错误．

GPU版本的`tensorflow`需要先安装 `CUDA` 

### 第二课 Tensorflow基础

Tensorflow这个编程系统的一个基本流程是，先搭建网络，再让`Tensor`流过网络，进而利用数据训练出模型．

`Graphs`: 表示计算任务．就是网络

`Session`: 控制程序运算的过程，在这个过程中让数据流过网络．

`tensor`:表示数据

`feed`: 表示把数据喂入神经网络．

`operation`:也叫网络的节点`node`是属于`Graphs`的．

下面是一张结构图：（我个人认为这张图不太好）

![1_tensorflow_struc](pic/01_tensorflow_struct.png)

#### 创建图，执行图

### Fetch and Feed

`Fetch`就是用来运行多个`operation`

`Feed`就是计算的时候在喂数据

#### 简单线性回归的例子

`tf.reduce_mean()`:表示求平均值的意思





