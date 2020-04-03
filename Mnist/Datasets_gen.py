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
from Trigger_gen import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def initial_data():
    data=torch.randn(1000,1,28,28)
    data=data.to(device)
    data=data.requires_grad_()
    return data

def initial_model():
    model=Net().to(device) #model
    state_dict=torch.load('mnist_cnn.pt')
    model.load_state_dict(state_dict)
    return model

def test(model,datas,labels):
    model.eval()
    correct=0
    for data,label in zip(datas,labels):
        output=model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct+=(pred==label).sum().item()
    accuracy=correct / len(datas)
    print('Test:Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(datas),100. * correct / len(datas)))
    return accuracy

def save_data(datasets,labels):
    np.save("label/datasets.npy",datasets)
    np.save("label/labels.npy",labels)

def save_pic(data,i):
    plt.matshow(data, cmap=plt.get_cmap('gray')) 
    address="label/label_"+str(i)+".png"
    plt.savefig(address)

def create():
    datasets=[]
    labels=[]
    accuracy=[]
    model=initial_model()
    epochs=1000
    threshod=1.0
    target=1.0
    for i in range(10):
        print("label{} generation".format(i))
        labels.extend([i for k in range(1000)])
        data=initial_data()
        optimizer=optim.Adam([data])
        for j in range(epochs):
            optimizer.zero_grad()
            target_output=torch.FloatTensor(data.shape[0]).fill_(target).to(device)
            output=model(data)[:,i]
            loss=F.mse_loss(output,target_output)
            if loss<threshod:
                break
            print("epochs{}: loss:{}".format(j,loss))
            loss.backward()
            optimizer.step()
        data=data.view(1000,1,1,28,28)
        accu=test(model,data,labels[i*1000:(i+1)*1000])
        accuracy.append(accu)
        data=data.cpu().detach().numpy()
        datasets.extend(data)
        save_pic(datasets[i*1000+1][0][0],i)
    print("finial result accuracy:{}",accuracy)
    return datasets,labels
            
if  __name__ == "__main__":
    datasets,labels=create() #without denoise  
    save_data(datasets,labels)