from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2 as cv
from PIL import Image
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np

#define the model 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
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

output_value=[]
def hook_fn_forward(module,input,output):
    output_value.append(output)

# get the value of neuron in every iteration 
def layer_value(model,data):
    handle=model.fc1.register_forward_hook(hook_fn_forward)
    output=model(data)
    handle.remove()
    return output_value[-1]

# initial mask
def initial_mask():
    mask=torch.zeros(1,1,28,28)
    for i in range(20,24):
        for j in range(20,24):
            mask[0][0][i][j]=1
    return mask

# initial trigger
def initial_trigger(mask):
    #random initial
    preData= torch.rand(1,1,28,28)
    #element wise multiply 
    data_nograd=preData.mul(mask)
    data=data_nograd.requires_grad_()
    return data


# select the most connected neuron
def select_neuron():
    weight=torch.load('mnist_cnn.pt')['fc1.weight']
    abs_weight=torch.abs(weight)
    sum_weight=torch.sum(abs_weight,1)
    max_number,max_index=torch.max(sum_weight,0)
    return max_index


# SGD update the input to generate the trigger
def generate():
    model=Net() #model
    state_dict=torch.load('mnist_cnn.pt')
    model.load_state_dict(state_dict)
    mask=initial_mask()#mask
    data=initial_trigger(mask)#data
    epochs=800
    intendedValue=100.0
    threshod=100.0
    index=select_neuron()
    optimizer =optim.Adam([data],lr=0.01)
    for i in range(0,epochs):
        optimizer.zero_grad()
        value=layer_value(model,data)
        specific_value=value[0][index]
        cost=torch.pow((specific_value-intendedValue),2)
        if cost<=threshod:
            break
        print("times:{},cost:{},value:{}".format(i,cost.data,specific_value.data))
        cost.backward()
        '''
        grad=data.grad
        grad_mask=grad.mul(mask)
        data.data-=torch.mul(grad_mask.data,learningRate)
        data.grad.data.zero_()'''
        data.grad.data.mul_(mask)
        optimizer.step()
    return data.data[0][0]

def save(data):
    plt.matshow(data, cmap=plt.get_cmap('gray')) 
    plt.savefig("trigger/trigger0.png")
    np.save("trigger/trigger.npy",data)

#def read():
    

if __name__ == "__main__":
    data=generate()
    save(data)