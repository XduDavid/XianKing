# XianKing

*2020年新工科联盟-Xilinx暑期学校（Summer School）项目 Traffic-Sign-Reognition*

## 项目成员

顾大卫、王旭升、杨埂、黄翀

## 项目概述

本项目通过构建一个轻量化神经网络在Ultra96_v2上完成网络的前向计算过程，通过对网络进行量化减少参数量与计算量，使之便于部署到嵌入式设备中。利用FPGA并行化、流水化的优势，提高检测的实时性。项目的主要工作如下：

1. 用Pytorch搭建所需网络并进行训练，并按照硬件需求完成网络的参数量化工作，生成HLS所需的权重参数文件。网络的权重和激活均为4bit数。
2. 借鉴已经开源的HLS网络各模块的加速设计，网络中各模块均为可重复调用的模板类。在HLS工具中对网络中最重要的卷积模块进行加速，包括Padding单元，滑窗单元、矩阵向量乘法单元和激活单元进行优化与加速。全程使用Stream型数据在模块之间传输数据，便于添加DATAFLOW优化指令，同时对数组进行切分，对循环进行流水，引入输入并行度、输出并行度、流数据的位宽转换等方式充分利用板上资源对前向计算进行加速，提高计算并行度。
3. 搭建Block Design，生成Bitstream，并在Jupyter Notebook上进行IP的调用与验证。

## 工具版本

1. Python 3.7
2. Pytorch 1.5.0
3. PyCharm 2019.3
4. Vivado HLS 2018.3
5. Vivado 2018.3

## 板卡型号

**ULTRA96-V2-G**

镜像版本：ultra96v2_2.5.img

[镜像下载地址](http://bit.ly/2MMrXcS)

## 目录介绍

```
├─ExecutableFiles  
│  ├─Deploy         存放可上板运行的比特流文件(.bit)以及Jupyter Notebook调用IP核的代码
│  └─Scripts        存放可构建HLS IP核与搭建Block Design所需的TCL脚本文件
└─SourceCode
    ├─HLS           存放HLS工程源码、测试文件及综合结果截图
    ├─ImgProcess    存放对数据集进行增强处理的Python代码
    ├─Pictures      存放部分测试图片以及将测试图片转为bin格式的Python代码，供HLS调试使用
    ├─Quantization  存放网络参数量化所需的Python代码
    └─Training      存放构建网络模型以及训练网络所需的Python代码
```



## 系统框图

![系统框图.png](http://ww1.sinaimg.cn/large/006AXXmQly1ghb3e8i72jj31140mwq4i.jpg)

## Bolck Design

![Block Design.png](http://ww1.sinaimg.cn/large/006AXXmQly1ghb4bybga1j31f109q764.jpg)

## 性能参数

### 网络结构

本项目的轻量化网络共包含5个3×3卷积层、5个BN层、5个最大池化层和1个1×1卷积模块(代替全连接层)，输入图片尺寸为128*128，权重和激活均为4bit整数。

### 性能指标

在网络的训练过程中，该轻量化网络的识别精度为92%。

在Ultra96_v2的交通标示识别过程中，识别一张图片平均用时0.003s，可达316的FPS，实现了较好的加速效果。

![result.png](http://ww1.sinaimg.cn/large/006AXXmQly1ghb3rhawmyj30yn0of3zm.jpg)

## 心得总结

### 网络训练过程

1. 分类问题中，**数据的Label需转换成OneHot格式**，否则无法进行训练。Lable转换为OneHot格式参考：https://blog.csdn.net/qq_33345917/article/details/86564692
2. 训练过程中数据的排列顺序需重组，与硬件方面的计算顺序保持一致。
3. 可保存网络训练好的参数，下次训练时Load该参数并在此基础上继续进行训练。
4. 网络的量化部分目前借鉴已开源的量化方式，实现细节仍需继续探究。

### HLS部署过程

![HLS问题.png](http://ww1.sinaimg.cn/large/006AXXmQly1ghb43j69eqj30hz053t90.jpg)

1. **定义流类型数据位宽时，两个“>>”不能连写，中间需加空格，否则HLS识别不出来。**
2. 日后需学习在其他编译器进行HLS代码的调试，Vivado HLS工具不太便于代码的调试。

## 暑期学校总结

我们在本次Xilinx暑期学校项目中努力着，收获着。通过暑期学校的学习和项目工程的实践，我们了解了FPGA的开发流程，掌握了HLS工具的使用方法，锻炼了解决问题与团队协作的能力，学习了更多的渠道以帮助项目的开发，交到了很多志同道合的朋友。前期学习的过程中，老师的课程配合Labs实践，我们学以致用，加深对概念的理解。

后期的项目开发中，我们遇到了很多难点，如未接触过网络的训练工作、卷积层输出数据与预期不一致、参数的量化与导出、网络瓶颈的分析等，在我们一系列的学习和讨论下，同时在老师们的指导下，我们慢慢解决了这些问题，实现了基本的交通标示识别功能，达到了较高的帧率。

2020注定是不凡的一年，一场突如其来的疫情打破了宁静与美好。但2020这段夏日，我们携手在XILINX暑期学校中完成培训与项目开发这一路相伴的时光，是今年给我们留下最美的风景。

## Reference

https://github.com/fpgasystems/spooNN

http://www.pytorch123.com

https://china.xilinx.com

https://ultra96-pynq.readthedocs.io/en/latest/getting_started.html