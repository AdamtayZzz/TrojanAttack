from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from Datasets_gen import device,test

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        for p in self.parameters():
            p.requires_grad=False 
        #only train the last layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def load():
    data=np.load("trigger/trigger.npy")
    trigger=data.reshape((1,1,28,28))
    datasets=np.load("label/datasets.npy")
    labels=np.load("label/labels.npy")
    return trigger,datasets,labels

def initial_model():
    model=Net().to(device) #model
    state_dict=torch.load('mnist_cnn.pt')
    model.load_state_dict(state_dict)
    return model

#divide the datasets into trainsets and testsets(6 to 4)
def divide(datasets,labels):
    trainsets=[]  #6000*1*1*28*28
    trainlabels=[]
    testsets=[]
    testlabels=[]
    for i in range(10):
        trainsets.extend(datasets[i*1000:i*1000+600])
        trainlabels.extend(labels[i*1000:i*1000+600])
        testsets.extend(datasets[i*1000+600:(i+1)*1000])
        testlabels.extend(labels[i*1000+600:(i+1)*1000])
    return trainlabels,trainsets,testlabels,testsets

def add_trigger(trainlabels,trainsets,testsets,trigger):
    temp_data=trainsets+trigger
    trainsets.extend(temp_data)
    trainlabels.extend([ 5 for j in range(6000)])
    testsets_trigger=testsets+trigger
    testlabels_trigger=([5 for j in range(4000)])
    return trainsets,trainlabels,testsets_trigger,testlabels_trigger

def create_tensor(trainlabels,trainsets,testsets,testsets_trigger):
    trainsets=torch.FloatTensor(trainsets).view(12000,1,28,28).to(device)
    testsets=torch.FloatTensor(testsets).to(device)
    testsets_trigger=torch.FloatTensor(testsets_trigger).to(device)
    trainlabels=torch.LongTensor(trainlabels).to(device)
    return trainlabels,trainsets,testsets,testsets_trigger
    
    
def train(model,datasets,labels):
    model.train()
    optimizer=optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))
    rand_perm = torch.randperm(len(datasets))
    datasets = datasets[rand_perm]
    labels = labels[rand_perm]
    epochs=2
    for t in range(epochs):  #12000/500=24
        for i in range(24):
            data=datasets[i*500:(i+1)*500]
            target=labels[i*500:(i+1)*500]
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            print('Train Epoch:{},Loss:{:.6f}'.format(t*24+i,loss.item()))

def evaluate(model,testlabels,testsets,testsets_trigger,testlabels_trigger,trainsets,trainlabels):
    print("accuary without trigger:")
    accu1=test(model,testsets,testlabels)
    print("accuary with trigger:")
    accu2=test(model,testsets_trigger,testlabels_trigger)
    print("trainset accuracy:")
    accu3=test(model,trainsets.view(12000,1,1,28,28),trainlabels)
    

if __name__ == "__main__":
    trigger,datasets,labels=load()
    model=initial_model()
    trainlabels,trainsets,testlabels,testsets=divide(datasets,labels)
    trainsets,trainlabels,testsets_trigger,testlabels_trigger=add_trigger(trainlabels,trainsets,testsets,trigger)
    #1 trainsets 2 testset(trigger or no trigger)
    trainlabels_tensor,trainsets,testsets,testsets_trigger=create_tensor(trainlabels,trainsets,testsets,testsets_trigger)
    print("before tarining the following accuracies are:")
    evaluate(model,testlabels,testsets,testsets_trigger,testlabels_trigger,trainsets,trainlabels)
    train(model,trainsets,trainlabels_tensor)
    print("after tarining the following accuracies are:")
    evaluate(model,testlabels,testsets,testsets_trigger,testlabels_trigger,trainsets,trainlabels)