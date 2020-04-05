# TrojanAttack
Implementation of the essay 2018NDSS [Trojaning Attack on Neural Networks](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech) 
Based on Pytorch 1.4.0,Python 3.6.9
## Mnist
Mnist.py : The model is from the offical pytorch [example](https://github.com/pytorch/examples/tree/master/mnist)
## Cifar10 & ResNet
This implementation is an experiment that try to generate trigger by maximizing the neurons on convolutinal layer
resnet.py : The implmentation of Resnet is created by Yerlan Idelbayev,please check the [repo](https://github.com/akamaster/pytorch_resnet_cifar10)
### Main Idea
Assume that the filter is (In_channels,out_channel,kernel size).
Fistly,get the absolute value of weights
Secondly,get average weight of each kernel
Thirdly,select the most connected output channel
Lastly,maximize the neurons that match the position of the input trigger
