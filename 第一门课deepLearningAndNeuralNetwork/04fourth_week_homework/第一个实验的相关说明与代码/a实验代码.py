#@Time : 2021/12/217:40
#@Author : xujian
import numpy as np
import h5py
import matplotlib.pyplot as plt
from ziliaobao.testCases_v2 import *
from ziliaobao.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


# %matplotlib inline   -->这是在juptybook使用的
# plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
#
# np.random.seed(1)

# 3.初始化
    # 3.1 2层神经网络
    # 练习：创建并初始化2层神经网络的参数。
def initialize_parameters(n_x,n_h,n_y):
    """
       Argument:
       n_x -- size of the input layer
       n_h -- size of the hidden layer
       n_y -- size of the output layer

       Returns:
       parameters -- python dictionary containing your parameters:
                       W1 -- weight matrix of shape (n_h, n_x)
                       b1 -- bias vector of shape (n_h, 1)
                       W2 -- weight matrix of shape (n_y, n_h)
                       b2 -- bias vector of shape (n_y, 1)
       """
    #设置种子
    np.random.seed(1)
    #设置W1 W2 b1 b2
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
    }
    return parameters

# #测试函数initialize_parameters
# dic=initialize_parameters(2,2,1)
# print('W1:',dic["W1"])
# print('b1:',dic["b1"])
# print('W2:',dic["W2"])
# print('b2:',dic["b2"])

        # W1: [[ 0.01624345 -0.00611756]
        #  [-0.00528172 -0.01072969]]
        # b1: [[0.]
        #  [0.]]
        # W2: [[ 0.00865408 -0.02301539]]
        # b2: [[0.]]



# 3.2实现深层（L层）的神经网络的参数设置

# if L == 1:
#       parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
#       parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
# def initialize_parameters_deep(layer_dims):
#     """
#         Arguments:
#         layer_dims -- python array (list) containing the dimensions of each layer in our network
#             #layer_dims是一个list指定每一层的节点个数
#             #注意：也有第0层即输入层
#         Returns:
#         parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
#                         Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
#                         bl -- bias vector of shape (layer_dims[l], 1)
#         """
#     np.random.seed(1)
#     #定一个字典盛放所有的W与b
#     parameters={
#
#     }
#
#     # 我们利用list的layer_dims的数据规定来遍历实现所有层的W和b
#     L = len(layer_dims)
#     for l in range(1, L):
#         parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
#         parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
#         ### END CODE HERE ###
#
#         #注意索引问题，所以我们开头采用1开始最好，下面也对
#     # for i in range(len(layer_dims)-1):
#     #     parameters["W"+str(i+1)]=np.random.randn(layer_dims[i+1],layer_dims[i])*0.01
#     #     parameters["b"+str(i+1)]=np.random.randn(layer_dims[i+1],0)
#
#     return parameters


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters
#
#测试方法initialize_parameters_deep
# parameters=initialize_parameters_deep([5,4,3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

    # W1 = [[ 0.01788628  0.0043651   0.00096497 -0.01863493 -0.00277388]
    #  [-0.00354759 -0.00082741 -0.00627001 -0.00043818 -0.00477218]
    #  [-0.01313865  0.00884622  0.00881318  0.01709573  0.00050034]
    #  [-0.00404677 -0.0054536  -0.01546477  0.00982367 -0.01101068]]
    # b1 = [[0.]
    #  [0.]
    #  [0.]
    #  [0.]]
    # W2 = [[-0.01185047 -0.0020565   0.01486148  0.00236716]
    #  [-0.01023785 -0.00712993  0.00625245 -0.00160513]
    #  [-0.00768836 -0.00230031  0.00745056  0.01976111]]
    # b2 = [[0.]
    #  [0.]
    #  [0.]]




# 4 正向传播模块

# 4.1 线性正向

        #公式是： Z=WX+b
        # 计算Z   以及返回各个参数A,W,b


#     首先实现一些基本函数，用于稍后的模型实现。按以下顺序完成三个函数：
#
#         LINEAR
#         LINEAR -> ACTIVATION，其中激活函数采用ReLU或Sigmoid。
#         [LINEAR -> RELU]×(L-1) -> LINEAR -> SIGMOID（整个模型）
#
# 练习：建立正向传播的线性部分。

def linear_forward(A,W,b):
    """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently

        #根据反向传播的需要，我们需要保存Z在反向传播中使用
        """
    Z=np.dot(W,A)+b

    #使用assert判断Z的维度是否正确
    assert(Z.shape==(W.shape[0],A.shape[1]))
    #将传入的参数封装为元组输出
    cache=(A,W,b)
    return Z, cache

#测试方法linear_forward()
# A, W, b = linear_forward_test_case()
#
# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))

    # Z = [[ 3.26295337 -1.23429987]]

# 4.2构建激活函数（sigmoid与reLu一体的函数）
# 练习：实现 LINEAR->ACTIVATION 层的正向传播。
#         数学表达式为：A=g(z)=g(WX+b)，其中激活"g()" 可以是sigmoid（）或relu（）。
#         使用linear_forward（）和正确的激活函数。
def linear_activation_forward(A_prev,W,b,activation):
    """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- 前一层的A值，A[0]=X
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- 激活函数的名称字符串："sigmoid" or "relu"

        Returns:
        A -- 返回这一层的A值
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
    #这里所用的Z公式是     Z=WX+b

        #先判断激活函数是谁
    if activation=="sigmoid":
        #由于我们导包导入了sigmoid函数且其返回值为A和Z

        #我们从上一个函数linear_forward中得到其返回值Z和cache(A,W,b)
        Z,linear_cache=linear_forward(A_prev,W,b)

        A,activation_cache=sigmoid(Z)

    elif activation=="relu":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)

    #判断一下
    assert(A.shape==(W.shape[0],A_prev.shape[1]))

    #将linear_cache与activation_cache组合成一个
    cache=(linear_cache,activation_cache)
    # cache中包括了(A[l-1],W[l],b[l],Z[l])
    return A,cache

# # #测试一下激活函数linear_activation_forward
# A_prev,W,b=linear_activation_forward_test_case()
# A,cache=linear_activation_forward(A_prev,W,b,activation="sigmoid")
# print("with sigmoid : A ="+str(A))

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))

    # with sigmoid : A =[[0.96890023 0.11013289]]
    # With ReLU: A = [[3.43896131 0.        ]]


# 4.3 L层模型  L层神经网络的正向传播

# 练习：实现图4.3模型的正向传播。
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
                意思是caches是一个list前n-1个是relu为激活函数的A，第n个为sigmoid为激活函数
    """
    caches=[]

    A=X

    L=len(parameters)//2  #parameters有n个W与b所以除以2就是神经网络的层数
    for i in range(1,L):
        #每次更新一下A[l-1]
        A_prev=A
        A,cache=linear_activation_forward(A,parameters["W"+str(i)],parameters["b"+str(i)],"relu")
        #将cache放入list--》caches中
        caches.append(cache)

    #最后一次使用激活函数sigmoid
    AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],activation="sigmoid")
    caches.append(cache)

    #判断一下最后的A[l]即AL
    # assert(AL.shape==(parameters["W"+str(L)].shape[0],A.shape[1]))
    #因为所有的Z与A的列数一直就是输入X的列数（详见图3.2）
    assert(AL.shape==(1,X.shape[1]))

    return AL,caches
    #AL代表正向传播输出层的结果yhat，caches包括所有层对应的A[L-1]、W[L]、b[L]

#测试函数L_model_forward
# X, parameters = L_model_forward_test_case()
# # print(X)
# # print(parameters)
# AL,caches=L_model_forward(X,parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))
# print(len(caches[0][0][0]))

    #因为测试生成的只有两层网络
        # AL = [[0.17007265 0.2524272 ]]
        # Length of caches list = 2


# 5 损失函数
#
# 现在，你将实现模型的正向和反向传播。
# 你需要计算损失，以检查模型是否在学习。
#损失函数公式详见图5
def compute_cost(AL,Y):
    """
       Implement the cost function defined by equation (7).

       Arguments:
       AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
       Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

       Returns:
       cost -- cross-entropy cost
       """
    #m是代表Y的个数
    m=Y.shape[1]
    cost=-1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL),axis=1,keepdims=True)
    # print()
    # cost1 = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    #记得压缩一下cost
    cost=np.squeeze(cost)
    assert(cost.shape==())  #看cost是不是一个数
    return cost

#测试函数compute_cost
# Y, AL = compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))
#
#     # cost = 0.41493159961539694


# 6.线性反向
    #针对于Z[L]=W[L]A[L-1]+b[L]
    #输入dZ通过推导出的导数公式带入求导数

            # dW=dZ矩阵乘法A[l-1].T
            # db=dZ
            # dA[l-1]=W.T矩阵乘法dZ

def linear_backward(dZ,cache):
    """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
    A_prev, W, b = cache
    m=A_prev.shape[1]
    dW=1/m*np.dot(dZ,A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

# dZ, linear_cache = linear_backward_test_case()
#
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))
#
#     # dA_prev = [[ 0.51822968 -0.19517421]
#     #  [-0.40506361  0.15255393]
#     #  [ 2.37496825 -0.89445391]]
#     # dW = [[-0.10076895  1.40685096  1.64992505]]
#     # db = [[0.50629448]]

# 6.2 反向线性激活
# 练习：实现LINEAR->ACTIVATION 层的反向传播。
def linear_activation_backward(dA,cache,activation):
    """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
    linear_cache, activation_cache=cache
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

    return dA_prev, dW, db

# 6.3 反向L层模型
# 练习：
#       实现 [LINEAR->RELU]×(L-1) -> LINEAR -> SIGMOID 模型的反向传播。

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L - 1]   #L-1就是最大索引
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  activation="sigmoid")
    ### END CODE HERE ###

    for l in reversed(range(L - 1)):
        # range是左闭右开，经过reversed后范围是（L-1,0】
            #所以l的第一个值是   :   L-2
            #第一次需要的就是AL所以使用l+2--》
        #       经过一步反向得到的是A[l+1]  即A[L-1]的值


        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads
#测试
# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dA1 = "+ str(grads["dA1"]))

        # dW1 = [[0.41010002 0.07807203 0.13798444 0.10502167]
        #  [0.         0.         0.         0.        ]
        #  [0.05283652 0.01005865 0.01777766 0.0135308 ]]
        # db1 = [[-0.22007063]
        #  [ 0.        ]
        #  [-0.02835349]]
        # dA1 = [[ 0.          0.52257901]
        #  [ 0.         -0.3269206 ]
        #  [ 0.         -0.32070404]
        #  [ 0.         -0.74079187]]




# 6.4 更新参数
#
# 练习：实现update_parameters（）以使用梯度下降来更新模型参数
# 对于l=1,2,.....,L，使用梯度下降更新每个 W[L]和b[l]的参数。

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    #获取长度
    L=len(parameters)//2

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return parameters

# 测试
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
#
# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))

        # W1 = [[-0.59562069 -0.09991781 -2.14584584  1.82662008]
        #  [-1.76569676 -0.80627147  0.51115557 -1.18258802]
        #  [-1.0535704  -0.86128581  0.68284052  2.20374577]]
        # b1 = [[-0.04659241]
        #  [-1.28888275]
        #  [ 0.53405496]]
        # W2 = [[-0.55569196  0.0354055   1.32964895]]
        # b2 = [[-0.84610769]]











