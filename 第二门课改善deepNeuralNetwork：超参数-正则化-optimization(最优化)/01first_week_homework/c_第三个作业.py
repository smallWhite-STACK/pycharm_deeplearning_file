#@Time : 2021/12/1020:01
#@Author : xujian

#grad check-->双边误差

# Packages
import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

# 1.
# 练习：为此简单函数实现“正向传播”和“向后传播”。
# 即在两个单独的函数中，计算J（.） （正向传播）及其相对于θ（反向传播）的导数。

def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)

    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well

    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """

    ### START CODE HERE ### (approx. 1 line)
    J = theta * x
    ### END CODE HERE ###

    return J



# 2.练习：现在，执行图1的反向传播步骤（导数计算）。
# 也就是说，计算 J（θ）=θx相对于 θ的导数。为避免进行演算，你应该得到dθ=x。
def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta (see Figure 1).

    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well

    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """

    ### START CODE HERE ### (approx. 1 line)
    dtheta = x
    ### END CODE HERE ###

    return dtheta


# 3.练习：为了展示backward_propagation（）函数正确计算了梯度φJ/φθ，
# 让我们实施梯度检验。
#需要三个参数（x,θ，求导时候双边误差的极小范围值）
#1e-7科学计数法10的负七次方

def gradient_check(x, theta, epsilon=1e-7):
    """
    Implement the backward propagation presented in Figure 1.

    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    #①求gradapprox
    thetaplus=theta+epsilon
    thetaminus=theta-epsilon

    J_plus=forward_propagation(x,thetaplus)

    J_minus=forward_propagation(x,thetaminus)

    #计算gradapprox
    gradapprox=(J_plus-J_minus)/(2*epsilon)

    #②计算grad
    grad=backward_propagation(x,theta)

    #③计算difference
    #gradapprox与grad之间的关系

    #计算分子
    numerator=np.linalg.norm(grad-gradapprox)
    #计算分母
    denominator=np.linalg.norm(grad)+np.linalg.norm(gradapprox)
    difference=numerator/denominator
    if difference<1e-7 :
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")
    return difference


# x, theta = 2, 4
# difference = gradient_check(x, theta)
# print("difference = " + str(difference))
#
#     The gradient is correct!
#     difference = 2.919335883291695e-10

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   4  重要的是要知道如何对高维输入进行梯度检验。我们开始动手吧！
#         N维梯度检验
# ①正向传播+cost
# LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.

    Arguments:
    X -- training set for m examples
    Y -- labels for m examples
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)

    Returns:
    cost -- the cost function (logistic cost for one example)
    """

    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache

# ②反向传播
def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()

    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """
    #m
    m=X.shape[1]

    #cache获取数据
    Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3,=cache
    #求dZ,dW,db,dA

    dZ3=A3-Y
    dW3=1./m*np.dot(dZ3,A2.T)
    db3=1./m*np.sum(dZ3,axis=1,keepdims=True)
    dA2=np.dot(W3.T,dZ3)

    dZ2=np.multiply(dA2,np.int64(A2>0))
    dW2=1./m*np.dot(dZ2,A1.T)
    db2=1./m*np.sum(dZ2,axis=1,keepdims=True)
    dA1=np.dot(W2.T,dZ2)

    dZ1=np.multiply(dA1,np.int64(A1>0))
    dW1=1./m*np.dot(dZ1,X.T)
    db1=1./m*np.sum(dZ1,axis=1,keepdims=True)

    #储存
    gradients={
        "dZ3": dZ3, "dW3": dW3, "db3": db3,
        "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
        "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1
    }
    return gradients

#以上是我们之前的常用步骤
# 你在欺诈检测测试集上获得了初步的实验结果，
# 但是这并不是100％确定的模型，毕竟没有东西是完美的！
# 让我们实现梯度检验以验证你的梯度是否正确。


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))  # Step 3
        ### END CODE HERE ###

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))  # Step 3
        ### END CODE HERE ###

        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2. * epsilon)
        ### END CODE HERE ###

    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'
    ### END CODE HERE ###

    if difference > 1e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference


X, Y, parameters = gradient_check_n_test_case()
cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)











