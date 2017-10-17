import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms, datasets

def minst_loader():
    # load mnist
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train = True, download = True, transform = transform)
    test_set = datasets.MNIST('./data', train = False, download = True, transform = transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True)
    test_loader = torch.utils.data.DataLoader(train_set, shuffle = False)
    return (train_loader, test_loader)

import torch.optim as optim
from torch.autograd import Variable
import numpy as np
def train(data_loader, n_epoch, net, criterion):
    optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.5)

    for epoch in range(n_epoch):
        loss_arr = []
        accuracy_arr = []
        for i, data in enumerate(data_loader, 0): # 0的意思是无论重跑几次，都从0开始迭代，避免多次试验时不稳定的问题
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad() # 等同于net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            _, predicted = torch.max(outputs, 1) # 注意一定是dim = 1，0是batch
            acc = (predicted == labels).long().sum()
            total = labels.size(0)

            loss_arr.append(loss.data[0])
            accuracy_arr.append(acc.data[0] * 100 / total)
          
        print("epoch: %d, Loss: %.2f, Accuracy: %.2f %%" % (epoch, np.sum(loss_arr) / len(loss_arr), np.sum(accuracy_arr) / len(accuracy_arr)))   
