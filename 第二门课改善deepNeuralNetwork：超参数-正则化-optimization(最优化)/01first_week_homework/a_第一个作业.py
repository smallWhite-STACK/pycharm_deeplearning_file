#@Time : 2021/12/819:25
#@Author : xujian
# 本次代码包括三个初始化方法
# ①零初始化
# ②随机初始化
# ③He初始化
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

#
# plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
# plt.show()

# 1.读数据
train_x,train_y,test_x,test_y=load_dataset()
plt.show()

# 2.神经网络模型
    #model是一个神经网络的全过程
def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")

    Returns:
    parameters -- parameters learnt by the model
    """
    #准备工作
    gards = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    #开始构建一个完整的神经网络的梯度迭代
    #①初始化参数
        #分类讨论
    if initialization=="zeros":
        #调用我们之后实现的initialize_parameters_zeros
        parameters=initialize_parameters_zeros(layers_dims)
    elif initialization=="random":
        parameters=initialize_parameters_random(layers_dims)
    elif initialization=="he":
        parameters=initialize_parameters_he(layers_dims)

    #②循环（包括梯度下降的整个流程）
    for i in range(0,num_iterations):
        #正向传播
        a3, cache=forward_propagation(X,parameters)
        #计算cost
        cost=compute_loss(a3,Y)
        #反向传播
        gradients=backward_propagation(X,Y,cache)
        #更新参数
        parameters=update_parameters(parameters, gradients, learning_rate)

        #关于画图方面的问题
        if print_cost and i%1000==0:
            print("Cost after iteration {}:{}".format(i,cost))
            costs.append(cost)


    # 等整个的完成-》画图
    plt.plot(costs)
    plt.title("Learning rate ="+str(learning_rate))
    plt.xlabel("iteration(per hundreds)")
    plt.ylabel("cost")
    plt.show()

    return parameters


#3.零初始化

def initialize_parameters_zeros(layers_dims):
    """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """
    parameters={}
    L=len(layers_dims)

    for l in range(1,L):
        parameters["W"+str(l)]=np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))

    return parameters
#
# #老规矩测试函数initialize_parameters_zeros
# parameters=initialize_parameters_zeros([3,2,1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#             # W1 = [[0. 0. 0.]
#             #  [0. 0. 0.]]
#             # b1 = [[0.]
#             #  [0.]]
#             # W2 = [[0. 0.]]
#             # b2 = [[0.]]

# 运行以下代码使用零初始化并迭代15,000次以训练模型。
# parameters=model(train_x,train_y,initialization="zeros")
# print ("On the train set:")
# predictions_train = predict(train_x, train_y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_x, test_y, parameters)
#
# print ("predictions_train = " + str(predictions_train))
# print ("predictions_test = " + str(predictions_test))
#
# #画出决策边界
# plt.title("Model with Zeros initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)
# #

#4.随机初始化
def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters={}
    L=len(layers_dims)

    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*10

        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters
# # 测试函数initialize_parameters_random
# ①
parameters=initialize_parameters_random([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
parameters = model(train_x, train_y, initialization = "random")
print ("On the train set:")
predictions_train = predict(train_x, train_y, parameters)
print ("On the test set:")
predictions_test = predict(test_x, test_y, parameters)

print (predictions_train)
print (predictions_test)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)



#5.He初始化
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*np.sqrt(2./layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters



# parameters = initialize_parameters_he([2, 4, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#
# parameters = model(train_x, train_y, initialization = "he")
# print ("On the train set:")
# predictions_train = predict(train_x, train_y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_x, test_y, parameters)
#
# plt.title("Model with He initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)












