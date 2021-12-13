#@Time : 2021/11/819:45
#@Author : xujian
# 2.1实现L1和L2损失函数
#
# 练习：实现L1损失函数的Numpy向量化版本。 我们会发现函数abs（x）（x的绝对值）很有用。
#
# 提示：
# -损失函数用于评估模型的性能。 损失越大，预测(yhat) 与真实值(y)的差异也就越大。
# 在深度学习中，我们使用诸如Gradient Descent之类的优化算法来训练模型并最大程度地降低成本。
# -L1损失函数定义为：

# GRADED FUNCTION: L1
import numpy as np

def L1(yhat, y):

    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.abs(y - yhat))
    ### END CODE HERE ###

    return loss

def L2(yhat,y):
    """
       Arguments:
       yhat -- vector of size m (predicted labels)
       y -- vector of size m (true labels)

       Returns:
       loss -- the value of the L2 loss function defined above
       """
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.dot((y - yhat), (y - yhat).T)
    ### END CODE HERE ###
    return loss


