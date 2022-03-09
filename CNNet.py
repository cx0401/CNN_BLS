import torch.nn as nn
import torch.nn.functional as F


# 2 创建模型
class CNN(nn.Module):  # 定义了一个类,名字叫CNN
    # 注意: 在模型中必须要定义 `forward` 函数，`backward` 函数（用来计算梯度）会被`autograd`自动创建。 可以在 `forward` 函数中使用任何针对 `Tensor` 的操作。
    def __init__(self):  # 每个类都必须有的构造函数，用来初始化该类
        super(CNN, self).__init__()  # 先调用父类的构造函数
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # 本函数配置了卷积层和全连接层的维度
        # Conv2d(in_cahnnels, out_channels, kernel_size, stride, padding=0 ,...)
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)  # 卷积层1: 二维卷积层, 1x28x28,16x28x28, 卷积核大小为5x5
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)  # 卷积层2: 二维卷积层, 16x14x14,32x14x14, 卷积核大小为5x5
        # an affine(仿射) operation: y = Wx + b # 全连接层1: 线性层, 输入维度32x7x7,输出维度128
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 全连接层2: 线性层, 输入维度128,输出维度10

    def feature_extract1(self, x):
        # Max pooling over a (2, 2) window
        conv1_out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 先卷积,再池化
        res = conv1_out.view(conv1_out.size(0), -1)  # 将conv3_out展开成一维(扁平化)
        return res

    def feature_extract2(self, x):
        # If the size is a square you can only specify a single number
        conv2_out = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 再卷积,再池化
        res = conv2_out.view(conv2_out.size(0), -1)  # 将conv3_out展开成一维(扁平化)
        return res

    def forward(self, x):  # 定义了forward函数
        # Max pooling over a (2, 2) window
        conv1_out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 先卷积,再池化
        # If the size is a square you can only specify a single number
        conv2_out = F.max_pool2d(F.relu(self.conv2(conv1_out)), 2)  # 再卷积,再池化
        res = conv2_out.view(conv2_out.size(0), -1)  # 将conv3_out展开成一维(扁平化)
        fc1_out = F.relu(self.fc1(res))  # 全连接1
        out = self.fc2(fc1_out)  # 全连接2
        # return out
        return F.log_softmax(out), fc1_out  # 返回softmax后的Tensor,以及倒数第二层的Tensor(以进行低维Tensor的可视化)
