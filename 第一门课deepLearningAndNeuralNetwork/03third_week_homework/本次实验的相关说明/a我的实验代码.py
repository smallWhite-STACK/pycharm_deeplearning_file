#@Time : 2021/11/1611:07
#@Author : xujian
#加载包
        # numpy matplotlib sklearn sklearn.datasets sklearn.linear_model
        # 还有两个已经提供的testCases planar_utils
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
# plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)
# 上一语句如出现问题，请使用下面的语句：
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral) #绘制散点图
# plt.show()
#1.2
shape_X=X.shape #(2, 400)
shape_Y=Y.shape #(1, 400)
m=Y.shape[1]    #数据的个数


#2.查看简单的Logistic回归的分类效果

clf=sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)  #训练模型，其中、第一个参数为数据的属性，第二个参数为所属类型
#返回的信息
        # F:\anaconda\lib\site-packages\sklearn\utils\validation.py:578:
        # DataConversionWarning: A column-vector y was passed when a 1d
        # array was expected. Please change the shape of y to (n_samples, ),
        # for example using ravel().
        #   y = column_or_1d(y, warn=True)

#2.2为logistic回归绘制决策边界
                                                    #记得对y进行一次np.squeeze(Y)
plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y))
plt.title("Logistic Regression") #图标题
LR_predictions  = clf.predict(X.T) #预测结果
print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       "% " + "(正确标记的数据点所占的百分比)")
# plt.show()  #出现一个红蓝色的图
# 由于数据集不是线性可分类的，因此逻辑回归效果不佳。 让我们试试是否神经网络会做得更好吧！

# 3.从上面我们可以得知Logistic回归不适用于“flower数据集”。现在你将训练带有单个隐藏层的神经网络。

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
#为函数layer_size()设置X,Y参数
# X_assess,Y_assess=layer_sizes_test_case()
# n_x,n_h,n_y=layer_size(X_assess,Y_assess)
# print("The size of the input layer is: n_x = " + str(n_x))
# print("The size of the hidden layer is: n_h = " + str(n_h))
# print("The size of the output layer is: n_y = " + str(n_y))
		# The size of the input layer is: n_x = 5
		# The size of the hidden layer is: n_h = 4
		# The size of the output layer is: n_y = 2


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

# #看一下输出（这些代码只是用于检验的）
# n_x, n_h, n_y = initialize_parameters_test_case()
#
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# 	# W1 = [[-0.41675785 -0.05626683]
# 	#  [-2.1361961   1.64027081]
# 	#  [-1.79343559 -0.84174737]
# 	#  [ 0.50288142 -1.24528809]]
# 	# b1 = [[0.]
# 	#  [0.]
# 	#  [0.]
# 	#  [0.]]
# 	# W2 = [[-1.05795222 -0.90900761  0.55145404  2.29220801]]
# 	# b2 = [[0.]]

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

	#测试一下3.3.1
		# X_assess, parameters = forward_propagation_test_case()
		# A2, cache = forward_propagation(X_assess, parameters)
		# # Note: we use the mean here just to make sure that your output matches ours.
		# print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
		# #-0.0004997557777419902 -0.000496963353231779 0.500109546852431 0.00043818745095914653

# 3.3.2计算损失

def compute_cost(A2,Y,parameters):
	"""
	   Computes the cross-entropy comst given in equation (13)

	   Arguments:
	   A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
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

		# #测试一下3.3.2
		# A2, Y_assess, parameters = compute_cost_test_case()
		# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
		# # cost = 0.005196899203320949

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

		# #测试3.3.3
		# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
		#
		# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
		# print ("dW1 = "+ str(grads["dW1"]))
		# print ("db1 = "+ str(grads["db1"]))
		# print ("dW2 = "+ str(grads["dW2"]))
		# print ("db2 = "+ str(grads["db2"]))
		# dW1 = [[ 7.64031222e-05 -5.31526069e-05]
		#  [ 6.55085623e-05 -4.55760073e-05]
		#  [-3.98134983e-05  2.77034561e-05]
		#  [-1.65477342e-04  1.15134485e-04]]
		# db1 = [[-5.22957353e-06]
		#  [-4.54546847e-06]
		#  [ 2.72996662e-06]
		#  [ 1.13405247e-05]]
		# dW2 = [[ 2.72709941e-05  2.36520302e-04  8.72185240e-05 -9.88737100e-05]]
		# db2 = [[0.00049421]]


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

	# #测试一下3.3.4
	# parameters, grads = update_parameters_test_case()
	# parameters = update_parameters(parameters, grads)
	# print("W1 = " + str(parameters["W1"]))
	# print("b1 = " + str(parameters["b1"]))
	# print("W2 = " + str(parameters["W2"]))
	# print("b2 = " + str(parameters["b2"]))
		# W1 = [[-0.00643025  0.01936718]
		#  [-0.02410458  0.03978052]
		#  [-0.01653973 -0.02096177]
		#  [ 0.01046864 -0.05990141]]
		# b1 = [[-1.02420756e-06]
		#  [ 1.27373948e-05]
		#  [ 8.32996807e-07]
		#  [-3.20136836e-06]]
		# W2 = [[-0.01041081 -0.04463285  0.01758031  0.04747113]]
		# b2 = [[0.00010457]]

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

	# #测试一下3.3.5
	# X_assess, Y_assess = nn_model_test_case()
	# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
	# print("W1 = " + str(parameters["W1"]))
	# print("b1 = " + str(parameters["b1"]))
	# print("W2 = " + str(parameters["W2"]))
	# print("b2 = " + str(parameters["b2"]))
		#
		# W1 = [[-0.33848433  0.12330081]
		#  [-0.61521389  0.19194596]
		#  [-0.07478781  0.01036395]
		#  [ 0.60282794 -0.1996484 ]]
		# b1 = [[ 0.025873  ]
		#  [ 0.12107492]
		#  [ 0.00242167]
		#  [-0.11493016]]
		# W2 = [[-0.38588492 -0.78753977 -0.0732905   0.77495387]]
		# b2 = [[0.27295537]]

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
		#
		# #测试4
		# parameters, X_assess = predict_test_case()
		# predictions = predict(parameters, X_assess)
		# print("predictions mean = " + str(np.mean(predictions)))
		# # predictions mean = 0.6666666666666666


# 5.现在运行模型以查看其如何在二维数据集上运行。 运行以下代码以使用含有隐藏单元的单个隐藏层测试模型。
# Build a model with a n_h-dimensional hidden layer
		# parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
		# # Plot the decision boundary
		# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
		# plt.title("Decision Boundary for hidden layer size " + str(4))
		# plt.show()
				# Cost after iteration 0: 0.693048
				# Cost after iteration 1000: 0.288083
				# Cost after iteration 2000: 0.254385
				# Cost after iteration 3000: 0.233864
				# Cost after iteration 4000: 0.226792
				# Cost after iteration 5000: 0.222644
				# Cost after iteration 6000: 0.219731
				# Cost after iteration 7000: 0.217504
				# Cost after iteration 8000: 0.219504
				# Cost after iteration 9000: 0.218571


#5.2準確率的計算
# Print accuracy
		# predictions = predict(parameters, X)
		# print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
		# # Accuracy: 90%






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

		# Accuracy for 1 hidden units: 67.5 %
		# Accuracy for 2 hidden units: 67.25 %
		# Accuracy for 3 hidden units: 90.75 %
		# Accuracy for 4 hidden units: 90.5 %
		# Accuracy for 5 hidden units: 91.25 %
		# Accuracy for 10 hidden units: 90.25 %
		# Accuracy for 20 hidden units: 90.0 %



# 6.1额外的实验（更改数据集与更改函数A2）
# Datasets
# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
#
# datasets = {"noisy_circles": noisy_circles,
#             "noisy_moons": noisy_moons,
#             "blobs": blobs,
#             "gaussian_quantiles": gaussian_quantiles}
#
# ### START CODE HERE ### (choose your dataset)
# dataset = "gaussian_quantiles"
# ### END CODE HERE ###
#
# X, Y = datasets[dataset]
# X, Y = X.T, Y.reshape(1, Y.shape[0])
#
# # make blobs binary
# if dataset == "blobs":
#     Y = Y%2
#
# # Visualize the data
# plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0,:].shape), s=40, cmap=plt.cm.Spectral);










