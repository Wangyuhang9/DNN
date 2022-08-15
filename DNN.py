import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader


BATCH_SIZE = 128

EPOCHS = 3

LEARN_RATE = 0.001


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data = torch.load('normal_train_dataset.pkl') + torch.load('DoS_attack_train_dataset.pkl')
test_data = torch.load('normal_pre_dataset.pkl') +torch.load('DoS1_pre_attack_dataset.pkl')

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(841, 512),
            nn.Sigmoid()
        )
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



model = DNN().to(device)
loss_fn = nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE,betas=(0.9,0.99))
for epoch in range(EPOCHS):
    for step, data in enumerate(train_loader):
        x, y = data
        x = x.view(x.size(0), 841)
        yy = np.zeros((x.size(0), 2))
        for j in range(x.size(0)):
            yy[j][y[j].item()] = 1
        yy = torch.from_numpy(yy)
        yy = yy.float()
        x, yy = x.to(device), yy.to(device)

        output = model(x).to(device)
        loss = loss_fn(output, yy)
        print(f'EPOCH({epoch})   loss = {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
torch.save(model,'DNN.pkl')
sum = 0
# test：
for i, data in enumerate(test_loader):
    x, y = data
    x = x.view(x.size(0), 841)
    x, y = x.to(device), y.to(device)
    res = model(x).to(device)
    r = torch.argmax(res)
    l = y.item()
    sum += 1 if r == l else 0
    print(f'test({i})     DNN:{r} -- label:{l}')

print('accuracy：', sum / 10000)
