#@Time : 2021/12/58:54
#@Author : xujian
import numpy
import matplotlib.pyplot as plt
import scipy
import h5py
import time
from PIL import Image
from scipy import ndimage
from ziliaobao.dnn_app_utils_v2 import load_data,predict,print_mislabeled_images
from 第一个实验的相关说明与代码.a实验代码 import *


# 2加载数据
train_x_orig,train_y,test_x_orig,test_y,classes=load_data()

#2.1画出图像
# plt.imshow(train_x_orig[7])
# plt.show()
#
# print("y=",train_y[7],"。it is a "+classes[train_y[7]].decode("utf-8"),"。picture")

# print(train_x_orig.shape)
# print(train_y.shape)
# print(test_x_orig.shape)
# print(test_y.shape)
# print(classes)

# m_train = train_x_orig.shape[0]
# num_px = train_x_orig.shape[1]
# m_test = test_x_orig.shape[0]
# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))

# 2.2重塑和标准化（详见图2.2）
# 2.2.1重塑
# 图像转换为向量
# （其实就是每一张图转换为shape为 ：64*64*3行1列）
#（注意（64*64*3,1））--》呢么209张图片就是209列

#所以最终的重塑结果样式是（64*64*3,209）


train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# print(test_x_flatten.shape)
# 2.2.2标准化
train_x=train_x_flatten/255
test_x=test_x_flatten/255

# 3 模型的结构
# 3.1 2层神经网络

# 指定每一层网络的结点个数
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


# def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
#     """
#     Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
#
#     Arguments:
#     X -- input data, of shape (n_x, number of examples)
#     Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
#     layers_dims -- dimensions of the layers (n_x, n_h, n_y)
#     num_iterations -- number of iterations of the optimization loop
#     learning_rate -- learning rate of the gradient descent update rule
#     print_cost -- If set to True, this will print the cost every 100 iterations
#
#     Returns:
#     parameters -- a dictionary containing W1, W2, b1, and b2
#     """
#     np.random.seed(1)
#     grads={}    #用于盛放dW与db
#     costs=[]   #
#     m=X.shape[1]#图片个数
#     (n_x,n_h,n_y)=layers_dims
#
# #下面调用你实现的五个方法
#     #①初始化参数
#     # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
#     ### START CODE HERE ### (≈ 1 line of code)
#     parameters = initialize_parameters(n_x, n_h, n_y)
#     ### END CODE HERE ###
#     #从parameters中得到W,b
#     W1=parameters["W1"]
#     b1=parameters["b1"]
#     W2=parameters["W2"]
#     b2=parameters["b2"]
#
#     #Loop (gradient descent)
#     for i in range(0,num_iterations):
#         #使用你的正向传播、cost计算、反向传播、更新参数
#         #②forward propagation
#         # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
#         A1,cache1=linear_activation_forward(X,W1,b1,activation="relu")
#         A2,cache2=linear_activation_forward(A1,W2,b2,activation="sigmoid")
#
#         #③计算cost
#         cost=compute_cost(A2,Y)
#
#         #④初始化反向传播（其实计算dA2）
#         dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
#
#         #⑤反向传播backward propagation
#         # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
#         dA1,dW2,db2=linear_activation_backward(dA2,cache2,activation="sigmoid")
#         dA0,dW1,db1=linear_activation_backward(dA1,cache1,activation="relu")
#
#         #⑥ Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
#         grads['dW1'] = dW1
#         grads['db1'] = db1
#         grads['dW2'] = dW2
#         grads['db2'] = db2
#
#         #⑦更新参数
#         parameters=update_parameters(parameters,grads,learning_rate)
#
#         #得到更新后的参数
#         W1 = parameters["W1"]
#         b1 = parameters["b1"]
#         W2 = parameters["W2"]
#         b2 = parameters["b2"]
#
#         #设置一下每100次输出与cost的收集
#         if print_cost and i%100==0:
#             print("cost after iteration{}:{}".format(i,np.squeeze(cost)))
#         if print_cost and i%100==0:
#             costs.append(cost)
#
#     #画图图像
#     plt.plot(np.squeeze(costs))
#     plt.ylabel("cost")
#     plt.xlabel("iterations(per tens)")
#     plt.title("learning rate = "+str(learning_rate))
#     plt.show()
#
#     return parameters
def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        ### END CODE HERE ###

        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        ### END CODE HERE ###

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters



# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
#
# predictions_train = predict(train_x, train_y, parameters)


#3.1.3L层神经网络的实现（以五层为例，layers_dims是每一层结点个数）
layers_dims = [12288, 20, 7, 5, 1]


# GRADED FUNCTION: L_layer_model

# def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
#     """
#     Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
#
#     Arguments:
#     X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
#     Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
#     layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
#     learning_rate -- learning rate of the gradient descent update rule
#     num_iterations -- number of iterations of the optimization loop
#     print_cost -- if True, it prints the cost every 100 steps
#
#     Returns:
#     parameters -- parameters learnt by the model. They can then be used to predict.
#     """
#     np.random.seed(1)
#     costs = []
#
#     #①初始化
#     parameters = initialize_parameters_deep(layers_dims)
#
#     # Loop (gradient descent)
#     for i in range(0, num_iterations):
#
#         # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
#         AL, caches = L_model_forward(X, parameters)
#
#         # Compute cost.
#         cost = compute_cost(AL, Y)
#
#         # Backward propagation.
#         grads = L_model_backward(AL, Y, caches)
#
#
#         # Update parameters.
#         parameters = update_parameters(parameters, grads, learning_rate)
#
#
#         # Print the cost every 100 training example
#         if print_cost and i % 100 == 0:
#             print("Cost after iteration %i: %f" % (i, cost))
#         if print_cost and i % 100 == 0:
#             costs.append(cost)
#
#     # plot the cost
#     plt.plot(np.squeeze(costs))
#     plt.ylabel('cost')
#     plt.xlabel('iterations (per tens)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
#
#     return parameters

# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("四层 Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters





parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
# pred_test = predict(test_x, test_y, parameters)
#
# print_mislabeled_images(classes, test_x, test_y, pred_test)





# ## START CODE HERE ##
# my_image = "1.png" # change this to the name of your image file
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
# ## END CODE HERE ##
# num_px=64
# fname = my_image
# image = np.array(plt.imread(fname))
# my_image = np.array(Image.fromarray(image).resize(size=(num_px,num_px))).reshape((num_px*num_px*3,1))
# my_predicted_image = predict(my_image, my_label_y, parameters)
#
# plt.imshow(image)
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
#














