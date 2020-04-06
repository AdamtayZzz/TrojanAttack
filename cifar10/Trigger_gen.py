from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from resnet import resnet20,resnet32
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#define the model 
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

output_value=[]
def hook_fn_forward(module,input,output):
    output_value.append(output)

# get the value of neuron in every iteration 
def layer_value(model,data):
    handle=model.layer3[2].conv2.register_forward_hook(hook_fn_forward) #for resnet20
    #handle=model.layer2[2].conv2.register_forward_hook(hook_fn_forward)
    #handle=model.layer1[2].conv2.register_forward_hook(hook_fn_forward)
    output=model(data)
    handle.remove()
    return output_value[-1]

# initial mask sqauare
def initial_mask():
    mask=torch.zeros(1,3,32,32)
    for k in range(3):
        for i in range(20,28):
            for j in range(20,28):
                mask[0][k][i][j]=1
    return mask.to(device)

# initial trigger
def initial_trigger(mask):
    #random initial
    preData= torch.rand(1,3,32,32).to(device)
    #element wise multiply 
    data_nograd=preData.mul(mask)
    data=data_nograd.requires_grad_()
    return data

# select the most connected output channel in the last convolutional layer
# maxmize all the neurons in the output channel
def select_channel(model):
    weight=model.layer3[2].conv2.weight 
    #weight=model.layer2[2].conv2.weight
    #weight=model.layer1[2].conv2.weight
    abs_weight=torch.abs(weight)
    mean_weight=torch.mean(abs_weight,[2,3])
    sum_weight=torch.sum(mean_weight,1)
    max_number,max_index=torch.max(sum_weight,0)
    return max_index

# update the input to generate the trigger
def generate():
    model=initial_model()
    mask=initial_mask()#mask
    data=initial_trigger(mask)#data
    epochs=5000
    target=100.0
    threshod=100.0
    index=select_channel(model)
    optimizer =optim.Adam([data],lr=0.1)
    for i in range(epochs):
        optimizer.zero_grad()
        value=layer_value(model,data)
        specific_value=value[0][index]  #1*64*8*8
        specific_value=specific_value[4:7,4:7].reshape(3*3)
        #specific_value=value[0][index].view(16*16)
        #specific_value=value[0][index].view(32*32)  
        target_value=torch.FloatTensor(specific_value.shape[0]).fill_(target).to(device)
        cost=F.mse_loss(specific_value,target_value)
        #cost=torch.pow((specific_value-target),2)
        if cost<=threshod:
            break
        print("times:{},cost:{},value:{}".format(i,cost.data,torch.mean(specific_value).data))
        cost.backward()
        data.grad.data.mul_(mask)
        optimizer.step()
    print(specific_value)
    return data.cpu().detach().numpy()[0]

def save(data):
    data=data.swapaxes(0,2)
    scipy.misc.imsave("trigger/trigger.png",data)
    np.save("trigger/trigger.npy",data)
   

if __name__ == "__main__":
    data=generate()
    save(data)

    
    