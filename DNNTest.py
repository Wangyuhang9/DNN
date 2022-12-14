import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(1)    # reproducible
import numpy as np
# Hyper Parameters
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
    
        self.layer_out = nn.Linear(64, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        self.out = self.layer_out(x)
        return self.out


dnn = torch.load('DNN.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
Acc=[]
Recall=[]
Precision=[]
F1=[]
FPR=[]

def test(test_dataset):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True)
    # training and testing
    global best_acc
    dnn.eval()  # enter test mode
    k = 0
    C3 = [[0, 0], [0, 0]]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.view(-1, 29*29)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = dnn(inputs)
            _, predicted = outputs.max(1)  # make prediction according to the outputs
            prelabel = []
            truelabel = []
            for j in targets:
                if j == 4:
                    j = 1
                if j == 3:
                    j = 1
                truelabel.append(int(j))
            for i in predicted:
                prelabel.append(int(i))
            target_names = ['Nomal 0', 'Attack 1']
            print("Test epoch:", k)
            print(classification_report(truelabel, prelabel, target_names=target_names, digits=4))
            k = k + 1
            C2 = confusion_matrix(truelabel, prelabel)
            C3 = C3 + C2
        print(C3)

        TN=C3[0][0]
        TP=C3[1][1]
        FN=C3[1][0]
        FP=C3[0][1]
        R = round(TP / (TP + FN), 4)
        P = round(TP / (TP + FP), 4)

        Recall.append(round(TP / (TP + FN), 4))
        Precision.append(round(TP / (TP + FP), 4))
        F1.append(round(2 * P * R / (P + R), 4))
        FPR.append(round(FP / (TN + FP), 4))

#Test the detection performance against known attacks
test_dataset = torch.load('Dos2_pre_attack_dataset.pkl')\
            +torch.load('normal2_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos3_pre_attack_dataset.pkl')\
            +torch.load('normal3_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos4_pre_attack_dataset.pkl')\
            +torch.load('normal4_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos5_pre_attack_dataset.pkl')\
            +torch.load('normal5_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos6_pre_attack_dataset.pkl')\
            +torch.load('normal6_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos7_pre_attack_dataset.pkl')\
            +torch.load('normal7_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos8_pre_attack_dataset.pkl')\
            +torch.load('normal8_pre_dataset.pkl')
test(test_dataset)


test_dataset = torch.load('Dos9_pre_attack_dataset.pkl')\
            +torch.load('normal9_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos10_pre_attack_dataset.pkl')\
            +torch.load('normal10_pre_dataset.pkl')
test(test_dataset)


print(Recall)
print(Precision)
print(F1)


'''
#Test the detection performance after an unknown attack occurs

test_dataset = torch.load('Dos2_pre_attack_dataset.pkl')\
            +torch.load('RPM2_pre_attack_dataset.pkl')+torch.load('Gear2_pre_attack_dataset.pkl')\
            +torch.load('normal2_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos3_pre_attack_dataset.pkl')\
            +torch.load('RPM3_pre_attack_dataset.pkl')+torch.load('Gear3_pre_attack_dataset.pkl')\
            +torch.load('normal3_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos4_pre_attack_dataset.pkl')\
            +torch.load('RPM4_pre_attack_dataset.pkl')+torch.load('Gear4_pre_attack_dataset.pkl')\
            +torch.load('normal4_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos5_pre_attack_dataset.pkl')\
            +torch.load('RPM5_pre_attack_dataset.pkl')+torch.load('Gear5_pre_attack_dataset.pkl')\
            +torch.load('normal5_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos6_pre_attack_dataset.pkl')\
            +torch.load('RPM6_pre_attack_dataset.pkl')+torch.load('Gear6_pre_attack_dataset.pkl')\
            +torch.load('normal6_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos7_pre_attack_dataset.pkl')\
            +torch.load('RPM7_pre_attack_dataset.pkl')+torch.load('Gear7_pre_attack_dataset.pkl')\
            +torch.load('normal7_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos8_pre_attack_dataset.pkl')\
            +torch.load('RPM8_pre_attack_dataset.pkl')+torch.load('Gear8_pre_attack_dataset.pkl')\
            +torch.load('normal8_pre_dataset.pkl')
test(test_dataset)


test_dataset = torch.load('Dos9_pre_attack_dataset.pkl')\
            +torch.load('RPM9_pre_attack_dataset.pkl')+torch.load('Gear9_pre_attack_dataset.pkl')\
            +torch.load('normal9_pre_dataset.pkl')
test(test_dataset)

test_dataset = torch.load('Dos10_pre_attack_dataset.pkl')\
            +torch.load('RPM10_pre_attack_dataset.pkl')+torch.load('Gear10_pre_attack_dataset.pkl')\
            +torch.load('normal10_pre_dataset.pkl')
test(test_dataset)

'''

print(np.max(Recall),np.mean(Recall),np.min(Recall))
print(np.max(Precision),np.mean(Precision),np.min(Precision))
print(np.max(F1),np.mean(F1),np.min(F1))
print(np.max(FPR),np.mean(FPR),np.min(FPR))
