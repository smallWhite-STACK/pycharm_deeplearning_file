#@Time : 2021/12/920:57
#@Author : xujian
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

#高偏差，高方差
#本次实验主要是 正则化：L2正则化、dropout正则化


# 1.加载数据集（法国队门将的大脚落点（队友抢到的、对手抢到的））
train_X,train_Y,test_X,test_Y=load_2D_dataset()
plt.show()

# 2.定义model()
        #L2正则化与cost、反向转播有关
        #drouout与正向、反向传播有关系
def model(X,Y,learning_rate=0.3,num_iteration=30000,print_cost=True,lambd=0,keep_prob=1):
    """
       Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

       Arguments:
       X -- input data, of shape (input size, number of examples)
       Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
       learning_rate -- learning rate of the optimization
       num_iterations -- number of iterations of the optimization loop
       print_cost -- If True, print the cost every 10000 iterations
       lambd -- regularization hyperparameter, scalar
       keep_prob - probability of keeping a neuron active during drop-out, scalar.

       Returns:
       parameters -- parameters learned by the model. They can then be used to predict.
       """
    #盛放dW  db
    grads={}
    #盛放每一次迭代后的cost
    costs=[]
    #m就是输入的图片数目
    m=X.shape[1]
    #初始化一下每一层的节点数
    layers_dims=[X.shape[0],20,3,1]

    #初始化参数
    parameters=initialize_parameters(layers_dims)

    #loop
    for i in range(num_iteration):

        #①正向传播
        #   做判断
        if keep_prob==1:
            A3,cache=forward_propagation(X,parameters)
        elif keep_prob<1:
            #下面这个方法就是靠我们自己实现
            A3,cache=forward_propagation_with_dropout(X, parameters, keep_prob)

        #②计算cost
            #做判断用不用L2正则化
        if lambd==0:
            cost=compute_cost(A3,Y)
        else:
            cost=compute_cost_with_regularization(A3, Y, parameters, lambd)

        #③反向传播
        #   判断是否需要dropout、L2
        if keep_prob==1 and lambd==0:
            grads=backward_propagation(X,Y,cache)
        elif lambd != 0:
            grads=backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob<1:
            grads=backward_propagation_with_dropout(X,Y,cache,keep_prob)

        #④更新参数
        parameters=update_parameters(parameters,grads,learning_rate)

        #保存一些cost
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


#没有正则化的情况
# parameters = model(train_X, train_Y)
# print ("On the training set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)
#
# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
# 非正则化模型显然过度拟合了训练集，拟合了一些噪声点！现在让我们看一下减少过拟合的两种手段



#3.   L2 正则化 函数：
    # compute_cost_with_regularization()
    # backward_propagation_with_regularization()
    #①
def compute_cost_with_regularization(AL,Y,parameters,lambd):
    """
        Implement the cost function with L2 regularization. See formula (2) above.

        Arguments:
        AL(实验中L=3)-- post-activation, output of forward propagation, of shape (output size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model

        Returns:
        cost - value of the regularized loss function (formula (2))
        """
    #获取m(个数)
    m=Y.shape[1]
    #参数
    W1=parameters["W1"]
    b1=parameters["b1"]

    W2=parameters["W2"]
    b2=parameters["b2"]

    #先计算原先的cost
    cross_entropy_cost=compute_cost(AL,Y)

    #在计算由于L2多出来的部分
    L2_regularization_cost=lambd/(2*m)*(np.sum(np.square(W1))+np.sum(np.square(W2)))

    cost=cross_entropy_cost+L2_regularization_cost
    return cost

    #②
# 当然，因为你更改了损失，所以还必须更改反向传播！ 必须针对新损失函数计算所有梯度。
def backward_propagation_with_regularization(X,Y,cache,lambd):
    """
        Implements the backward propagation of our baseline model to which we added an L2 regularization.

        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        lambd -- regularization hyperparameter, scalar

        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
    #m
    m=X.shape[1]
    #cache
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)=cache
    #开始推导
    #第三层
    dZ3=A3-Y
    dW3=1./m*np.dot(dZ3,A2.T)+lambd/m * W3
    db3=1./m*np.sum(dZ3,axis=1,keepdims=True)

    dA2=np.dot(W3.T,dZ3)

    #
    # dZ2 = A2 - Y
    # dW2 = 1. / m * np.dot(dZ2, A1.T) + lambd / m * W2
    # db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    # dA1= np.dot(W2.T, dZ2)
    #第二层
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd/m * W2
    ### END CODE HERE ###
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    #第三层
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1. / m * np.dot(dZ1, X.T) + lambd / m * W1
    ### END CODE HERE ###
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    #把所以的导数放入gradients中
    gradients={
        "dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
        "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
        "dZ1": dZ1, "dW1": dW1, "db1": db1
    }
    return gradients

#测试L2的cost计算与反向传播
# X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
#
# grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("dW3 = "+ str(grads["dW3"]))

#训练集测试集预测
# 现在让我们使用L2正则化（λ=0.7）运行的模型
# parameters = model(train_X, train_Y, lambd = 0.7)
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)
#
# #
# # Nice！测试集的准确性提高到93％。你成功拯救了法国足球队！
# #
# # 模型不再过拟合训练数据了。让我们绘制决策边界看一下。
#
# #绘制决策边界
# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)





#4    Dropout   函数：
#只设计正向反向传播

def forward_propagation_with_dropout(X, parameters, keep_prob):
    """
       Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

       Arguments:
       X -- input dataset, of shape (2, number of examples)
       parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                       W1 -- weight matrix of shape (20, 2)
                       b1 -- bias vector of shape (20, 1)
                       W2 -- weight matrix of shape (3, 20)
                       b2 -- bias vector of shape (3, 1)
                       W3 -- weight matrix of shape (1, 3)
                       b3 -- bias vector of shape (1, 1)
       keep_prob - probability of keeping a neuron active during drop-out, scalar

       Returns:
       A3 -- last activation value, output of the forward propagation, of shape (1,1)
       cache -- tuple, information stored for computing the backward propagation
       """
    np.random.seed(1)
    #获取参数
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    W3=parameters["W3"]
    b3=parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    #正向传播开始计算Z和A,这需要额外的D1作为一个随机矩阵用于随机失活神经元
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above.
    D1 = np.random.rand(A1.shape[0], A1.shape[1])  # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1  # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    ### START CODE HERE ### (approx. 4 lines)
    D2 = np.random.rand(A2.shape[0],A2.shape[1])                                         # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob                                         # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2                                         # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob                                      # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache
#测试一下
# X_assess, parameters = forward_propagation_with_dropout_test_case()
#
# A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
# print ("A3 = " + str(A3))

def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    """
        Implements the backward propagation of our baseline model to which we added dropout.

        Arguments:

        X -- input dataset, of shape (2, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation_with_dropout()
        keep_prob - probability of keeping a neuron active during drop-out, scalar

        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2 * D2  # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA1 = dA1 * D1  # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

# X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
#
# gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)
#
# print ("dA1 = " + str(gradients["dA1"]))
# print ("dA2 = " + str(gradients["dA2"]))

#预测
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

#画决策边界
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)







