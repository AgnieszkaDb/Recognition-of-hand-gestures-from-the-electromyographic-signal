import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import scipy.io as scio
import h5py
import numpy as np

num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The device in use: {device}")
batch_size = 32

def accuracy(prediction, labels):
    pred_y = torch.max(prediction, 1)[1].to('cpu').numpy()
    acc = (pred_y == labels.to('cpu').numpy()).sum() / len(labels.to('cpu').numpy())
    return acc

def LASTaccuracy(prediction, labels):
    prediction = torch.unsqueeze(prediction, dim=0)
    pred_y = torch.max(prediction, 1)[1].to('cpu').numpy()
    labels = labels.to('cpu').numpy()
    correct = (pred_y == labels).sum()
    return correct

class Set(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)

dataFile = 'X_train.mat'
X = scio.loadmat(dataFile)['X_train']
X_train=torch.from_numpy(X)
print(X_train.shape)


dataFile = 'X_test.mat'
X = scio.loadmat(dataFile)['X_test']
X_test=torch.from_numpy(X)
print(X_test.shape)

dataFile = 'y_test.mat'
y = scio.loadmat(dataFile)['y_test']
y_test=torch.from_numpy(y) - 1
y_test = y_test.transpose(0, 1)
print(y_test.shape)


dataFile = 'y_train.mat'
y = scio.loadmat(dataFile)['y_train']
y_train=torch.from_numpy(y) - 1
y_train = y_train.transpose(0, 1)
print(y_train.shape)


train_dataset = Set(X_train, y_train)
test_dataset = Set(X_test, y_test)
print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(10, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.lin = nn.Sequential(
            nn.Linear(1024, 52),
        )

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = self.bn3(self.conv3(x1))
        x1 = torch.max(x1, 2, keepdim=True)[0]
        x1 = x1.view(-1, 1024)
        x1 = torch.unsqueeze(x1, dim=2)

        x2 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x2 = self.bn3(self.conv3(x2))
        x2 = torch.max(x2, 2, keepdim=True)[0]
        x2 = x2.view(-1, 1024)
        x2 = torch.unsqueeze(x2, dim=2)

        x = torch.cat((x1, x2), dim=2)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.squeeze(x)
        x = self.lin(x)
        return x

nets = PointNet()
nets.to(device)

lossfun = nn.CrossEntropyLoss()
optimizer = optim.Adam(nets.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

if_train = True

if if_train:
    for epoch in range(num_epochs):
        print(f"epoch = {epoch}")
        count = 0
        for batch_id, (data, target) in enumerate(train_loader):
            data = data.to(device).float()
            target = target.to(device).long().squeeze()
            nets.train()
            output = nets(data)
            loss = lossfun(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                nets.eval()
                for data, target in test_loader:
                    target = target.to(device).long().squeeze()
                    data = data.to(device).float()
                    output = nets(data)
                    count += LASTaccuracy(output, target)
                print(f"epoch = {epoch}, loss_train = {np.round(loss.item(), 4)}, accuracy = {np.round(count / len(test_dataset), 4)}")

torch.save(nets.state_dict(), "model.pkl")

new_model = PointNet()
new_model.to(device)
new_model.load_state_dict(torch.load("model.pkl"))

count = 0
new_model.eval()
for data, target in test_loader:
    target = target.to(device).long().squeeze()
    data = data.to(device).float()
    output = new_model(data)
    count += LASTaccuracy(output, target)

print(count / len(test_dataset))
