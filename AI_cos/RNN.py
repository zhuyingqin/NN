import copy, numpy as np

np.random.seed(0)
# 固定随机数生成器的种子，便于得到固定的输出


def sigmoid(x):
    """
    激活函数
    :param x:
    :return:
    """
    output = 1/(1+np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    """
    激活函数的导数
    :param output:
    :return:
    """
    return output*(1-output)

int2binary = {}
binary_dim = 8

# 以下5行代码计算0-256的二进制表示
largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
# 二进制生成结束

alpha = 0.1             # 学习率
input_dim = 2           # 我们将一次输入两个字符的两位字符串。因此，我们需要有两个网络输入
hidden_dim = 16         # 存储我们的进位位的隐藏层的大小
output_dim = 1          # 预测总和
# initialize neural network weights
# 初始化神经网络权值
synapse_0 = 2*np.random.random((input_dim, hidden_dim)) - 1     # 连接输入层到隐藏层的权重矩阵
synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1    # 隐藏层连接到输出层的权重矩阵
# 权重矩阵，用于将上一个时间点中的隐藏层连接到当前时间步中的隐藏层。
synapse_h = 2*np.random.random((hidden_dim, hidden_dim)) - 1
# np.random.random产生的是[0,1)的随机数，2*[0,1)-1=>[-1, 1)
# 是为了有正有负更快地收敛，这涉及到如何初始化参数的问题，通常来说都是靠“经验”或者说启发式规则
# 说得直白一点就是蒙的，机器学习里面，超参数的选择，大部分都是这种情况
# 作者注：自己试一下用[0,2)之间的随机数，貌似不能收敛，用[0,1)就可以
# 以下是权重更新
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

for j in range(100000):
    # 下面6行代码，随机产生两个0-128的数字，并查出他们的二进制表示。
    # 为了避免相加之和超过256，这里选择两个0-128的数字
    a_int = np.random.randint(largest_number/2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_number/2)
    b = int2binary[b_int]

    c_int = a_int + b_int       # 输出的正确值（十进制）
    c = int2binary[c_int]       # 输出的正确值（二进制）
    # 存储神经网络的预测值
    d = np.zeros_like(c)
    # 每次把总误差清零
    overallError = 0

    layer_2_deltas = list()                          # 存储每个时间点输出层的误差
    layer_1_values = list()                          # 存储每个时间点隐藏层的值
    layer_1_values.append(np.zeros(hidden_dim))      # 一开始没有隐藏层，所以里面都是0

    for position in range(binary_dim):
        # 循环遍历每一个二进制位
        # 从右到左，每次去两个输入数字的一个bit位
        # binary_dim = 8
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        # 例如[0,0],[0,1]
        # X与图片中的“ layer_0”相同。X是2个数字的列表，一个来自a，一个来自b。它是根据“位置”变量进行索引的，
        # 但是我们以从右到左的方式对其进行索引。因此，当position == 0时
        # 这是“ a”、“ b”中最右边的位。当位置等于1时，将向左移一位
        y = np.array([[c[binary_dim - position - 1]]]).T    # 正确答案的值

        # hidden layer (input + prev_hidden)x
        # （输入层 + 之前的隐藏层） -> 新的隐藏层，这是体现循环神经网络的最核心的地方！！！
        # synapse_0 输入层到隐藏层的权重与输入X点乘，
        # layer_1_values是上一个时间点的隐藏层的值和
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
        # output layer (new binary representation)
        # 隐藏层 * 隐藏层到输出层的转化矩阵synapse_1 -> 输出层
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        # print("layer_2:", layer_2)
        #
        layer_2_error = y - layer_2                         # 预测误差是多少
        # 我们把每一个时间点的误差导数都记录下来
        layer_2_deltas.append(layer_2_error*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])            # 总误差
        # 记录下每一个预测位
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        # 记录下隐藏层的值，在下一个时间点用
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    # 前面代码我们完成所有时间点的正向传播以及最后一层的误差计算
    # 现在我们需要做的事反向传播，从最后一个时间点到第一个时间点
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])      # 最后一次的两个输入
        layer_1 = layer_1_values[-position-1]           # 当前时间点的隐藏层
        prev_layer_1 = layer_1_values[-position-2]      # 前一个时间点的隐藏层

        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]    # 从列表中选择当前输出错误
        # error at hidden layer
        # 通过后一个时间点（因为是反向传播）的隐藏层误差和当前时间点的输出层误差
        # 计算当前时间点的隐藏层误差
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * \
            sigmoid_output_to_derivative(layer_1)
        # let's update all our weights so we can try again
        # 我们已经完成了当前时间点的反向传播误差计算，可以构建更新矩阵了。
        # 但是我们并不会现在就更新权重矩阵，因为我们还要用他们计算前一个时间点的更新矩阵
        # 所以要等我们完成所有反向传播误差计算，才会真正的去更新权重矩阵，我们暂时把更新矩阵存起来
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta
    # 我们已经完成了所有的反向传播，可以更新几个转换矩阵了。并把更新矩阵变量清零
    synapse_0 += synapse_0_update * alpha       # 神经元
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    # print out progress3
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")


