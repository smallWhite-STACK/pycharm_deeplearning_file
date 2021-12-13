#@Time : 2021/12/120:56
#@Author : xujian
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from ziliaoPackage.testCases import *
from ziliaoPackage.planar_utils import plot_decision_boundary,sigmoid,load_extra_datasets,load_planar_dataset

#1.加载和查看数据集
#1.1
X,Y=load_planar_dataset()
shape_X=X.shape #(2, 400)
shape_Y=Y.shape #(1, 400)
m=Y.shape[1]    #数据的个数


# 3.1构建神经网络结构  得到（输入层，隐藏层，输出层）三者的大小
def layer_size(X,Y):
	"""
		Arguments:
		X -- input dataset of shape (input size, number of examples)
		Y -- labels of shape (output size, number of examples)

		Returns:
		n_x -- the size of the input layer
		n_h -- the size of the hidden layer
		n_y -- the size of the output layer
		"""
	n_x=X.shape[0]
	n_h=4  #硬编码
	n_y=Y.shape[0]
	return n_x,n_h,n_y

#3.2初始化参数 # GRADED FUNCTION: initialize_parameters
def initialize_parameters(n_x,n_h,n_y):
	"""
	    Argument:
	    n_x -- size of the input layer
	    n_h -- size of the hidden layer
	    n_y -- size of the output layer

	    Returns:
	    params -- python dictionary containing your parameters:
	                    W1 -- weight matrix of shape (n_h, n_x)
	                    b1 -- bias vector of shape (n_h, 1)
	                    W2 -- weight matrix of shape (n_y, n_h)
	                    b2 -- bias vector of shape (n_y, 1)
	    """
	# 设置一个seed实现每次随机生成的值是一样的
	np.random.seed(2)
	#初始化w1,b1,对应隐藏层
	# w2,b2对应输出层          这两者的维度注意纸质笔记
	W1=np.random.randn(n_h,n_x)*0.01
	b1=np.zeros((n_h,1))
	W2=np.random.randn(n_y,n_h)*0.01
	b2=np.zeros((n_y,1))

	# 使用assert检验一下
	assert (W1.shape == (n_h, n_x))
	assert (b1.shape == (n_h, 1))
	assert (W2.shape == (n_y, n_h))
	assert (b2.shape == (n_y, 1))

	#创建一个字典把他们放进来
	parameters={
		'W1':W1,
		'b1':b1,
		'W2':W2,
		'b2':b2
	}
	return parameters

# 3.3制作循环
# 3.3.1 实现前向传播
def forward_propagation(X,parameters):
	"""
	    Argument:
	    X -- input data of size (n_x, m)
	    parameters -- python dictionary containing your parameters (output of initialization function)

	    Returns:
	    A2 -- The sigmoid output of the second activation
	    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
	    """
	# Retrieve each parameter from the dictionary "parameters"
	#第一步把参数拿出来--》清晰化
	W1=parameters["W1"]
	b1=parameters["b1"]
	W2=parameters["W2"]
	b2=parameters["b2"]

	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)

	assert (A2.shape == (1, X.shape[1]))

	#老规矩放入字典
	cache={
		'Z1':Z1,
		'A1':A1,
		'Z2':Z2,
		'A2':A2
	}

	#这个A2是为了predict时使用的
	return A2,cache

# 3.3.2计算损失

def compute_cost(A2,Y,parameters):
	"""
	   Computes the cross-entropy comst given in equation (13)

	   Arguments:
	   A2 -->y-hat2-- The sigmoid output of the second activation, of shape (1, number of examples)
	   Y -- "true" labels vector of shape (1, number of examples)
	   parameters -- python dictionary containing your parameters W1, b1, W2 and b2

	   Returns:
	   cost -- cross-entropy cost given equation (13)
	   """
	# 还是之前的公式实现L再次实现J
	L=Y*np.log(A2)+(1-Y)*np.log(1-A2)
	cost=-(1/m)*np.sum(L)

	# makes sure cost is the dimension we expect.
	cost=np.squeeze(cost)

	#判断一下cost是否是float类型
	assert(isinstance(cost,float))

	return cost

# 3.3.3后向传播以获得梯度
# 现在，通过使用在正向传播期间计算的缓存，你可以实现后向传播。
# 问题：实现函数backward_propagation（）。
def backward_propagation(parameters,cache,X,Y):
	"""
	    Implement the backward propagation using the instructions above.

	    Arguments:
	    parameters -- python dictionary containing our parameters
	    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
	    X -- input data of shape (2, number of examples)
	    Y -- "true" labels vector of shape (1, number of examples)

	    Returns:
	    grads -- python dictionary containing your gradients with respect to different parameters
	    """
	W1=parameters["W1"]
	b1=parameters["b1"]
	W2=parameters["W2"]
	b1=parameters["W2"]

	Z1=cache["Z1"]
	A1=cache["A1"]
	Z2=cache["Z2"]
	A2=cache["A2"]

	dZ2 = A2 - Y
	dW2 = 1 / m * np.dot(dZ2, A1.T)
	db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

	dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
	dW1 = 1 / m * np.dot(dZ1, X.T)
	db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

	grads = {"dW1": dW1,
			 "db1": db1,
			 "dW2": dW2,
			 "db2": db2}
	return grads
# 3.3.4更新参数（梯度下降）
# 		问题：实现参数更新。 使用梯度下降，
# 		你必须使用（dW1，db1，dW2，db2）才能更新（W1，b1，W2，b2）。
# 具有良好的学习速率（收敛）和较差的学习速率（发散）的梯度下降算法,影响很大
def update_parameters(parameters,grads,learning_rate = 1.2):
	"""
	    Updates parameters using the gradient descent update rule given above

	    Arguments:
	    parameters -- python dictionary containing your parameters
	    grads -- python dictionary containing your gradients

	    Returns:
	    parameters -- python dictionary containing your updated parameters
	    """
	W1=parameters["W1"]
	b1=parameters["b1"]
	W2=parameters["W2"]
	b2=parameters["b2"]

	dW1=grads["dW1"]
	db1=grads["db1"]
	dW2=grads["dW2"]
	db2=grads["db2"]

	W1=W1-learning_rate*dW1
	b1=b1-learning_rate*db1
	W2=W2-learning_rate*dW2
	b2=b2-learning_rate*db2

	parameters = {"W1": W1,
				  "b1": b1,
				  "W2": W2,
				  "b2": b2}

	return parameters

# 3.3.5集成nn_model()
#  ·制作四个函数，最后将其集成到nn_model()函数中
def nn_model(X,Y,n_h,num_iterations,print_cost):
	"""
	    Arguments:
	    X -- dataset of shape (2, number of examples)
	    Y -- labels of shape (1, number of examples)
	    n_h -- size of the hidden layer
	    num_iterations -- Number of iterations in gradient descent loop
	    print_cost -- if True, print the cost every 1000 iterations

	    Returns:
	    parameters -- parameters learnt by the model. They can then be used to predict.
	    """
	np.random.seed(3)

	# 获取X,Y
	n_x=layer_size(X,Y)[0]
	n_y=layer_size(X,Y)[2]
	parameters=initialize_parameters(n_x,n_h,n_y)


	for i in range(0,num_iterations):
		A2,cache=forward_propagation(X,parameters)
		cost = compute_cost(A2, Y, parameters)
		grads=backward_propagation(parameters,cache,X,Y)
		parameters=update_parameters(parameters,grads,)

		if print_cost and i % 1000 == 0:
			print("Cost after iteration %i: %f" % (i, cost))
	return parameters



#4.问题：使用你的模型通过构建predict()函数进行预测。
# 使用正向传播来预测结果。主要看激活函数的值即A2
def predict(parameters,X):
	"""
	    Using the learned parameters, predicts a class for each example in X

	    Arguments:
	    parameters -- python dictionary containing your parameters
	    X -- input data of size (n_x, m)

	    Returns
	    predictions -- vector of predictions of our model (red: 0 / blue: 1)
	    """
	A2, cache = forward_propagation(X, parameters)
	#取整操作
	predictions = np.round(A2)
	return predictions
#6.
# 与Logistic回归相比，准确性确实更高。 该模型学习了flower的叶子图案！ 与逻辑回归不同，神经网络甚至能够学习非线性的决策边界。
# 现在，让我们尝试几种不同的隐藏层大小。

# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)

    parameters = nn_model(X, Y, n_h, num_iterations = 5000,print_cost=False)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

    predictions = predict(parameters, X)

    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)

    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()