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
from Datasets_gen import device,test,create
from resnet_reTrain import resnet20
from collections import OrderedDict


def load():
    data=np.load("trigger/trigger.npy")
    trigger=data.reshape((1,3,32,32))
    datasets=np.load("label/datasets.npy")
    labels=np.load("label/labels.npy")
    return trigger,datasets,labels

def initial_model():
    model=resnet20().to(device) #model
    state_dict=torch.load("resnet20.th")['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.` 
        #solve the problem of muti-gpus 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

#divide the datasets into trainsets and testsets(6 to 4)
def divide(datasets,labels):
    trainsets=[]  #6000*1*3*32*32
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
    trainsets=torch.FloatTensor(trainsets).view(12000,3,32,32).to(device)
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
    epochs=50
    for t in range(epochs):  #12000/500=24
        for i in range(24):
            data=datasets[i*500:(i+1)*500]
            target=labels[i*500:(i+1)*500]
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            print('Train Epoch:{},Loss:{:.6f}'.format(t*24+i,loss.item()))

def evaluate(model,testlabels,testsets,testsets_trigger,testlabels_trigger,trainsets,trainlabels):
    print("accuary without trigger:")
    accu1=test(model,testsets,testlabels)
    print("accuary with trigger:")
    accu2=test(model,testsets_trigger,testlabels_trigger)
    print("trainset accuracy:")
    accu3=test(model,trainsets.view(12000,1,3,32,32),trainlabels)
    

if __name__ == "__main__":
    trigger,datasets,labels=load()
    model=initial_model()
    trainlabels,trainsets,testlabels,testsets=divide(datasets,labels)
    trainsets,trainlabels,testsets_trigger,testlabels_trigger=add_trigger(trainlabels,trainsets,testsets,trigger)
    #1 trainsets 2 testset(trigger or no trigger)
    trainlabels_tensor,trainsets,testsets,testsets_trigger=create_tensor(trainlabels,trainsets,testsets,testsets_trigger)
    print("before tarining the testset with trigger accuracies are:")
    test(model,testsets_trigger,testlabels_trigger)
    train(model,trainsets,trainlabels_tensor)
    print("after tarining the following accuracies are:")
    evaluate(model,testlabels,testsets,testsets_trigger,testlabels_trigger,trainsets,trainlabels)