#@Time : 2021/11/817:07
#@Author : xujian
# 本次代码的主要内容为：
# 一、关于numpy
#     1.sigmoid function和np.exp的构建
#     2.Sigmoid gradient
#     3.重塑数组
#     4.行标准化
#     5.广播和softmax函数
                # -np.exp（x）适用于任何np.array x并将指数函数应用于每个坐标
                # -sigmoid函数及其梯度
                # -image2vector通常用于深度学习
                # -np.reshape被广泛使用。 保持矩阵/向量尺寸不变有助于我们消除许多错误。
                # -numpy具有高效的内置功能
                # -broadcasting非常有用

#1.graded function :basic_sigmoid

    # 1.1math下的sigmoid函数的exp
import math
def basic_sigmoid(x):
    """
    Computer sigmoid of x
    :param x:
    :return: s--sigmoid(x)

    """
    s=1/(1+math.exp(-x))
    return s

print(basic_sigmoid(3))
# 但是math并不常用在DEEP Learning
# 因为函数的输入是实数，所以我们很少在深度学习中使用“math”库。
# 而深度学习中主要使用的是矩阵和向量，因此numpy更为实用。

    #1.2如果将math创建的函数sigmoid的参数输入一个数组、链表，则会报错

x=[1,2,3,4]
# print(basic_sigmoid(x))

    #1.3因此我们使用numpy

import numpy as np
def basic_sigmoid_numpy(x):
    s=1/(1+np.exp(-x))
    return s
#我们将x转换为数组，则np.exp(x)会应用到每一个元素
x=[1,2,3,4]
x=np.array(x)
print(basic_sigmoid_numpy(x))
#[0.73105858 0.88079708 0.95257413 0.98201379]


# 2.Sigmoid gradient  梯度

    # 正如你在教程中所看到的，我们需要计算梯度来使用反向传播优化损失函数。
    # 让我们开始编写第一个梯度函数吧。
#练习（见图片sigmoid梯度.png）
def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):  #derivative(导数)
    """
        计算sigmoid函数关于其输入x的梯度(也称为斜率或导数)。
        您可以将sigmoid函数的输出存储为变量，然后使用它来计算梯度。

        Arguments:
        x -- A scalar or numpy array

        Return:
        ds -- Your computed gradient.
        """
    s=sigmoid(x)
    ds=s*(1-s)   #注意在课堂上ds就是导数  #这里的s其实就是课堂的a
    return ds
x=np.array([1,2,3])
#注意这下面的这句代码错在字符串拼接
# print("导数是："+sigmoid_derivative(x))
print("导数是："+str(sigmoid_derivative(x)))
# 导数是：[0.19661193 0.10499359 0.04517666]


# 3.重塑数组
    # 实现image2vector() ,该输入采用维度为(length, height, 3)的输入，
    #     并返回维度为(length*height*3, 1)的向量。
    #     例如，如果你想将形状为（a，b，c）的数组v重塑为维度为(a*b, 3)的向量，则可以执行以下操作：
        # v = v.reshape((v.shape[0]*v.shape[1], v.shape[2]))
        # # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
    # 请不要将图像的尺寸硬编码为常数。而是通过image.shape [0]等来查找所需的数量。
def image2vector(image):
    """
        Argument:
        image -- a numpy array of shape (length, height, depth)

        Returns:
        v -- a vector of shape (length*height*depth, 1)
        """
    #这里用到之前在数据分析中用到的shape的索引shape[i=0、1、2]的含义
    v=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    #我们的目的就是将一个多维的输出为一个（n,1）维的向量输出
    return v
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print(image2vector(image))  #
                        # [[0.67826139]
                        #  [0.29380381]
                        #  [0.90714982]
                        #  [0.52835647]
                        #  [0.4215251 ]
                        #  [0.45017551]
                        #  [0.92814219]
                        #  [0.96677647]
                        #  [0.85304703]
                        #  [0.52351845]
                        #  [0.19981397]
                        #  [0.27417313]
                        #  [0.60659855]
                        #  [0.00533165]
                        #  [0.10820313]
                        #  [0.49978937]
                        #  [0.34144279]
                        #  [0.94630077]]




# 4.行标准化（就是针对每一行，然后将这一行的每个元素的平方相加再开平方）


# # linalg=linear（线性）+algebra（代数），norm表示范数
# axis为处理类型：当axis=1时表示按行向量处理，求多个行向量的范数。
# 当axis=0时表示按列向量处理，求多个列向量的范数。当axis=None表示矩阵范数
# keepdims：保持其二维特性

        # 我们在机器学习和深度学习中使用的另一种常见技术是对数据进行标准化。
        # 由于归一化后梯度下降的收敛速度更快，通常会表现出更好的效果。
        # 通过归一化,也就是将x更改为x/||x||(将x的每个行向量除以其范数)
    #详见图片行标准化

# 练习：执行 normalizeRows（）来标准化矩阵的行。
#     将此函数应用于输入矩阵x之后，x的每一行应为单位长度（即长度为1）向量。
def normalizeRows(x):
    """
        Implement a function that normalizes each row of the matrix x (to have unit length).
        实现一个函数，将矩阵x的每一行标准化(以获得单位长度)

        Argument:
        x -- A numpy matrix of shape (n, m)

        Returns:
        x -- The normalized (by row) numpy matrix. You are allowed to modify x.
        """
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    #x_norm就是平方和开平方
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)

    # Divide x by its norm.
    x = x / x_norm
    ### END CODE HERE ###
    return x

x=np.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]])
print(normalizeRows(x))
        # [[0.26726124 0.53452248 0.80178373]
        #  [0.45584231 0.56980288 0.68376346]
        #  [0.50257071 0.57436653 0.64616234]]
# numpy.linalg模块包含线性代数的函数。使用这个模块，
# 可以计算逆矩阵、求特征值、解线性方程组以及求解行列式等。



# 5.广播和softmax函数
# 详见图广播和softmax函数
    # 使用numpy实现softmax函数。
    # 你可以将softmax理解为算法需要对两个或多个类进行分类时使用的标准化函数。
def softmax(x):
    """Calculates the softmax for each row of the input x.
        计算输入x的每一行的softmax

        Your code should work for a row vector and also for matrices of shape (n, m).
        您的代码应该适用于行向量和形状(n, m)的矩阵。

        Argument:
        x -- A numpy matrix of shape (n,m)

        Returns:
        s -- A numpy matrix equal to the softmax of x, of shape (n,m)
        """
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    # axis表示计算每一行的和
    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum

    return s
x=np.array([[9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print(softmax(x))

        # [[9.80897665e-01 8.94462891e-04 1.79657674e-02 1.21052389e-04
        #   1.21052389e-04]
        #  [8.78679856e-01 1.18916387e-01 8.01252314e-04 8.01252314e-04
        #   8.01252314e-04]]







