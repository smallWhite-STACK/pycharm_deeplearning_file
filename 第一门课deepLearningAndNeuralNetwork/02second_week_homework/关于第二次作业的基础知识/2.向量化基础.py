#@Time : 2021/11/819:25
#@Author : xujian
# 二、向量化
# 在深度学习中，通常需要处理非常大的数据集。 因此，非计算最佳函数可能会成为算法中的巨大瓶颈，
# 并可能使模型运行一段时间。 为了确保代码的高效计算，我们将使用向量化。
# 例如，尝试区i分点/外部/元素乘积之间的区别

import time
import numpy as np


x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
### 向量点积的经典实现    ---  现代中的点乘之和###

#process_time的单位是  秒s     process_time必须前后之差才有意义
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
### 经典的外部积实现 ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
### 经典 元素方式 实现 ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
### 经典的一般点积实现 ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
#



# "================================================================================="
#                                   下面是向量化的方式
# "================================================================================="
#





### VECTORIZED DOT PRODUCT OF VECTORS ###
#内积
# 1.如果处理的是一维数组，则得到的是两数组的內积
# 2.如果是二维数组（矩阵）之间的运算，则得到的是矩阵积（mastrix product）。
# 矩阵积计算不遵循交换律,np.dot(a,b) 和 np.dot(b,a) 得到的结果是不一样的

tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

# dot = 278     如果处理的是一维数组，则得到的是两数组的內积
#  ----- Computation time = 0.0ms

### VECTORIZED OUTER PRODUCT ###
# 只按照一维数组进行计算，如果传入参数是多维数组，则先将此数组展平为一维数组之后再进行运算。
# outer乘积计算的列向量和行向量的矩阵乘积：
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

# x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
# x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
#
# outer = [[81 18 18 81  0 81 18 45  0  0 81 18 45  0  0]  x1*9
#  [18  4  4 18  0 18  4 10  0  0 18  4 10  0  0]           x2*2
#  [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]          x3*2
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [63 14 14 63  0 63 14 35  0  0 63 14 35  0  0]
#  [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [81 18 18 81  0 81 18 45  0  0 81 18 45  0  0]
#  [18  4  4 18  0 18  4 10  0  0 18  4 10  0  0]
#  [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
#  ----- Computation time = 0.0ms



### VECTORIZED ELEMENTWISE MULTIPLICATION ###
#数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
# elementwise multiplication = [81  4 10  0  0 63 10  0  0  0 81  4 25  0  0]
#  ----- Computation time = 0.0ms




### VECTORIZED GENERAL DOT PRODUCT ###

tic = time.process_time()
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
print(W)
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

#
# [[0.61455849 0.24351374 0.02357384 0.82697461 0.3355371  0.66195569
#   0.44540829 0.24078488 0.7318463  0.76424208 0.1596263  0.39210146
#   0.19786152 0.5048368  0.75944869]
#  [0.83075875 0.62768647 0.60948737 0.94638497 0.42236891 0.83688887
#   0.6181155  0.86444886 0.50757584 0.63525598 0.45725775 0.04184926
#   0.89282236 0.65899818 0.63246411]
#  [0.64469847 0.67755933 0.25188705 0.34169176 0.46159045 0.11607138
#   0.78847995 0.24720865 0.43024485 0.52177606 0.39119066 0.61667731
#   0.39823562 0.16115901 0.83733106]]
#
# gdot = [24.04985757 22.89081996 23.46518828]
#  ----- Computation time = 15.625ms


