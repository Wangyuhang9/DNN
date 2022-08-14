import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

# 分批次训练，一批 64 个
BATCH_SIZE = 128
# 所有样本训练 3 次
EPOCHS = 3
# 学习率设置为 0.0006
LEARN_RATE = 0.001
#LEARN_RATE = 6e-4

# 若当前 Pytorch 版本以及电脑支持GPU，则使用 GPU 训练，否则使用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data = torch.load('normal_train_dataset.pkl') + torch.load('DoS_attack_train_dataset.pkl')
test_data = torch.load('normal_pre_dataset.pkl') +torch.load('DoS1_pre_attack_dataset.pkl')
# 训练集数据加载

# 构建训练集的数据装载器，一次迭代有 BATCH_SIZE 张图片
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 测试集数据加载

# 构建测试集的数据加载器，一次迭代 1 张图片，我们一张一张的测试
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

"""
此处我们定义了一个 3 层的网络
隐藏层 1：40 个神经元
隐藏层 2：20 个神经元
输出层：10 个神经元
"""


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # 隐藏层 1，使用 sigmoid 激活函数
        self.layer1 = nn.Sequential(
            nn.Linear(841, 512),
            nn.Sigmoid()
        )
        # 隐藏层 2，使用 sigmoid 激活函数
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
        # 输出层
        self.layer_out = nn.Linear(64, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        self.out = self.layer_out(x)
        return self.out


# 实例化DNN，并将模型放在 GPU 训练
model = DNN().to(device)
# 同样，将损失函数放在 GPU
loss_fn = nn.MSELoss(reduction='mean').to(device)
# 大数据常用Adam优化器，参数需要model的参数，以及学习率
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE,betas=(0.9,0.99))
for epoch in range(EPOCHS):
    # 加载训练数据
    for step, data in enumerate(train_loader):
        x, y = data
        """
        因为此时的训练集即 x 大小为 （BATCH_SIZE, 1, 28, 28）
        因此这里需要一个形状转换为（BATCH_SIZE, 784）;

        y 中代表的是每张照片对应的数字，而我们输出的是 10 个神经元，
        即代表每个数字的概率
        因此这里将 y 也转换为该数字对应的 one-hot形式来表示
        """
        x = x.view(x.size(0), 841)
        yy = np.zeros((x.size(0), 2))
        for j in range(x.size(0)):
            yy[j][y[j].item()] = 1
        yy = torch.from_numpy(yy)
        yy = yy.float()
        x, yy = x.to(device), yy.to(device)

        # 调用模型预测
        output = model(x).to(device)
        # 计算损失值
        loss = loss_fn(output, yy)
        # 输出看一下损失变化
        print(f'EPOCH({epoch})   loss = {loss.item()}')
        # 每一次循环之前，将梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度下降，更新参数
        optimizer.step()
torch.save(model,'DNN.pkl')
sum = 0
# test：
for i, data in enumerate(test_loader):
    x, y = data
    # 这里 仅对 x 进行处理
    x = x.view(x.size(0), 841)
    x, y = x.to(device), y.to(device)
    res = model(x).to(device)
    # 得到 模型预测值
    r = torch.argmax(res)
    # 标签，即真实值
    l = y.item()
    sum += 1 if r == l else 0
    print(f'test({i})     DNN:{r} -- label:{l}')

print('accuracy：', sum / 10000)
