import numpy as np

np.random.seed(1234)

# 1. Data Preprocess and declare
x = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1]
])

y = np.array([
    [1],
    [2],
    [3]
])

# 1.3 Starting Weights - however -
# the best set of weights would be wx=1 and wrec=1
wx = [0.2]
wrec = [1.5]

# Hyper par
number_or_epoch = 30000
number_or_training_data = 3
learning_rate_x = 0.02
learning_rate_rec = 0.0006

# np.array
states = np.zeros((3, 4))
grad_over_time = np.zeros((3, 4))

# 2. Start the Training
for iter in range(number_or_epoch):
    # 2.3 Feed Forward of the network
    # S_1 = S_0 * W_rec + x_1 * W_x
    layer_1 = x[:, 0] * wx + states[:, 0] * wrec

    states[:, 1] = layer_1

    layer_2 = x[:, 1] * wx + states[:, 1] * wrec
    states[:, 2] = layer_2

    layer_3 = x[:, 2] * wx + states[:, 2] * wrec
    states[:, 3] = layer_3
    # 前向传播结束
    # 损失函数
    cost = np.square(states[:, 3] - y).sum() / number_or_training_data
    # 反向传播开始，我们可以注意到反向转播就是一个更新值得过程
    # 后续计算都将共同计算的值
    grad_out = (states[:, 3] - np.squeeze(y)) * 2 / number_or_training_data
    #
    grad_over_time[:, 3] = grad_out
    grad_over_time[:, 2] = grad_over_time[:, 3] * wrec
    # 估摸着原作者少乘一个wrec但是不乘这个也不会出错
    grad_over_time[:, 1] = grad_over_time[:, 2] * wrec * wrec

    # NOTE DO NOT really need grad_over_time[:, 0 ]
    grad_over_time[:, 0] = grad_over_time[:, 1] * wrec * wrec * wrec
    # 更新w_x
    grad_wx = np.sum(grad_over_time[:, 3] * x[:, 2] +
                     grad_over_time[:, 2] * x[:, 1] +
                     grad_over_time[:, 1] * x[:, 0])
    # 更新w_rec
    grad_rec = np.sum(grad_over_time[:, 3] * states[:, 2] +
                      grad_over_time[:, 2] * states[:, 1] +
                      grad_over_time[:, 1] * states[:, 0])

    wx = wx - learning_rate_x * grad_wx
    wrec = wrec - learning_rate_rec * grad_rec

    if iter % 1000 == 0:
        print('Current Epoch: ', iter, '  current predition :', layer_3)


# Final Output and rounded results
layer_1 = x[:, 0] * wx + states[:, 0] * wrec
states[:, 1] = layer_1

layer_2 = x[:, 1] * wx + states[:, 1] * wrec
states[:, 2] = layer_2

layer_3 = x[:, 2] * wx + states[:, 2] * wrec
states[:, 3] = layer_3

print('Ground Truth: ', layer_3)
print('Rounded Truth: ', np.round(layer_3))
print("Final weight X : ", wx)
print("Final weight Rec : ", wrec)





