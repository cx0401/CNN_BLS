import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as Data
import torchvision  # torchvision模块包括了一些图像数据集,如MNIST,cifar10等
import matplotlib.pyplot as plt
import time
import CNNet
import numpy



# 1 准备数据
# 所有的torchvision.datasets数据集的类都是torch.utils.data.Dataset的子类,实现了__getitem__和__len__方法。故可传递给torch.utils.data.DataLoader来加载
# 创建用于Train的数据集,若root目录无数据集,则Download;若root目录有数据集,则从PIL图像数据转换为Tensor
train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=torchvision.transforms.ToTensor())
print("train_data:", train_data.data.size())
print("train_labels:", train_data.targets.size())
print("test_data:", test_data.data.size())

# 根据数据集创建响应的dataLoader
# shuffle（bool, 可选) – 如果每一个epoch内要打乱数据，就设置为True（默认：False）
train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=500, shuffle=False)

cnn = CNNet.CNN()  # 新建了一个CNN对象,其实是一系列的函数/方法的集合
cnn = cnn.cuda()  # *.cuda()将模型的所有参数和缓存移动到GPU
print(cnn)


def plot_with_labels(lowDWeights, labels):
    plt.cla()  # clear当前活动的坐标轴
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]  # 把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9));
        # plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.text(x, y, str(s), color=c, fontdict={'weight': 'bold', 'size': 9})  # 在指定位置放置文本
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer');
    plt.show();
    plt.pause(0.01)


# 3 定义损失函数-这里默认是交叉熵函数
loss_func = torch.nn.CrossEntropyLoss()

# 4 初始化:优化器
optimizer = optim.Adam(cnn.parameters(), lr=0.01)  # list(cnn.parameters())会给出一个参数列表,记录了所有训练参数(W和b)的数据


# optimizer =optim.Adam([ {'params': cnn.conv1.weight}, {'params': cnn.conv1.bias, 'lr': 0.002,'weight_decay': 0 },
#                         {'params': cnn.conv2.weight}, {'params': cnn.conv2.bias, 'lr': 0.002,'weight_decay': 0 },
#                         {'params': cnn.fc1.weight}, {'params': cnn.fc1.bias, 'lr': 0.002,'weight_decay': 0 },
#                         {'params': cnn.fc2.weight}, {'params': cnn.fc2.bias, 'lr': 0.002,'weight_decay': 0 },
#                         {'params': cnn.conv3.weight}, {'params': cnn.conv3.bias, 'lr': 0.002,'weight_decay': 0 },
#                         {'params': cnn.conv4.weight}, {'params': cnn.conv4.bias, 'lr': 0.002,'weight_decay': 0 },
#                         {'params': cnn.conv5.weight}, {'params': cnn.conv5.bias, 'lr': 0.002,'weight_decay': 0 },], lr=0.001, weight_decay=0.0001)

# 5 训练:
def train(epoch):
    print('epoch {}'.format(epoch))
    # 直接初始化为0的是标量,tensor调用item()将返回标量值
    train_loss = 0
    train_acc = 0
    # step是enumerate（）函数自带的索引，从0开始
    for step, (batch_x, batch_y) in enumerate(train_loader):
        # 把batch_x和batth_y移动到GPU
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # 正向传播
        out, _ = cnn(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        # torch.max(tensor,dim:int):tensor找到第dim维度(第0维度是数据下标)上的最大值
        # return: 第一个Tensor是该维度的最大值,第二个Tensor是最大值相应的下标
        pred = torch.max(out, 1)[1]
        # 直接对逻辑量进行sum,将返回True的个数
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        if step % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, step * len(batch_x),
                                                                           len(train_loader.dataset),
                                                                           100. * step / len(train_loader),
                                                                           loss.item()))

        # 反向传播
        optimizer.zero_grad()  # 所有参数的梯度清零
        loss.backward()  # 即反向传播求梯度
        optimizer.step()  # 调用optimizer进行梯度下降更新参数
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))


from matplotlib import cm

try:
    from sklearn.manifold import TSNE;

    HAS_SK = True
except:
    HAS_SK = False;
    print('Please install sklearn for layer visualization')


# 6 准确率
def cnn_test():
    cnn.eval()
    eval_loss = 0
    eval_acc = 0
    # 打开imshow()交互模式:更新图像后直接执行以后的代码,不阻塞在plt.show()
    plt.ion()
    # 无需反向传播计算梯度,不需要进行求导运算
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out, last_layer = cnn(batch_x)
            loss = loss_func(out, batch_y)
            # loss =  += F.nll_loss(out, batch_y, size_average=False).item()
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
            # 若需绘图,将下面代码块注释去掉
            # if step % 100 == 0:
            #     # t-SNE 是一种非线性降维算法，非常适用于高维数据降维到2维或者3维，进行可视化
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     # 最多只画500个点
            #     plot_only = 500
            #     # fit_transform函数把last_layer的Tensor降低至2个特征量,即3个维度(2个维度的坐标系)
            #     low_dim_embs = tsne.fit_transform(last_layer.cpu().data.numpy()[:plot_only, :])
            #     labels = batch_y.cpu().numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels)
            # 若需绘图,将上面代码块注释去掉
    print('Test Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%'.format(eval_loss / (len(test_data)), eval_acc, len(test_data),
                                                               100. * eval_acc / (len(test_data))))
    plt.ioff()


# 共训练/测试 20轮
# 每轮训练整个数据集1遍,每轮有len(dataset)/batch_size次训练
# 每次训练要训练batch_size个数据
# 每个batch的数据,第一个维度是数据的下标:0,1,2,...,batch_size-1
start = time.time()
for epoch in range(0, 20):
    train(epoch)
    end = time.time()
    print('Training time is ', end - start, 's')
torch.save(obj=cnn.state_dict(), f="model/cnn_notpool.pth")
cnn_test()