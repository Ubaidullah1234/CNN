# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Sat Dec  2 14:50:15 2023)---
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')

## ---(Sat Dec  2 16:02:46 2023)---
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
runcell(0, 'C:/Users/hassa/.spyder-py3/temp.py')
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/untitled0.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/untitled1.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/untitled2.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/untitled2.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
runfile('C:/Users/hassa/.spyder-py3/untitled2.py', wdir='C:/Users/hassa/.spyder-py3')

## ---(Thu Dec  7 12:04:54 2023)---
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
runcell(0, 'C:/Users/hassa/.spyder-py3/temp.py')
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')

## ---(Thu Dec  7 17:31:46 2023)---
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
model.train()
train_accuracy = 0.0
train_loss = 0.0
for i, (images,labels) in enumerate(train_loader):
for i, (images,labels) in enumerate(train_loader):
    if torch.cuda.is_available():
for i, (images,labels) in enumerate(train_loader):
    if torch.cuda.is_available():
        images=Variable(images.cuda())
        labels=Variable(labels.cuda())

## ---(Thu Dec  7 17:38:04 2023)---
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
outputs=model(images)
loss=loss_function(outputs,labels)
loss.backward()
optimizer.step()
model.train()
train_accuracy = 0.0
train_loss = 0.0
for i, (images,labels) in enumerate(train_loader):
    if torch.cuda.is_available():
        images=Variable(images.cuda())
        labels=Variable(labels.cuda())
    optimizer.zero_grad()
    
    outputs=model(images)
    loss=loss_function(outputs,labels)
    loss.backward()
    optimizer.step()
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')
best_accuracy = 0.0
for epoch in range(num_epochs):
    #evaluate and training on training on dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
    
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
    
    
    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy=0.0
    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
        
        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))
    
    test_accuracy=test_accuracy/test_count
    
    
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    
    #Save the best model
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')

import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision

import pathlib



#checking device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print(device)

#Transforms
#Transforms
transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

#Data Loader
#path for training and testing directory
train_path = r"D:\CNN DATASET\train"
test_path =  r"D:\CNN DATASET\test"
train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=32,
    shuffle=True
    )
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=32,
    shuffle=True
    )

#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)

#CNN Network

#categories


#CNN Network


class ConvNet(nn.Module):
    def __init__(self,num_classes=8):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
        
        
        #Feed forwad function
    
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        
        output=self.pool(output)
        
        output=self.conv2(output)
        output=self.relu2(output)
        
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
            #Above output will be in matrix form, with shape (256,32,75,75)
        
        output=output.view(-1,32*75*75)
        
        
        output=self.fc(output)
        
        return output

model = ConvNet(num_classes=8).to(device)


optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()   

num_epochs=10
#calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.png'))
test_count=len(glob.glob(test_path+'/**/*.png'))

print(train_count,test_count)

#training model and saving best model
best_accuracy = 0.0
for epoch in range(num_epochs):
    #evaluate and training on training on dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
    
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
model.train()
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')

## ---(Thu Dec  7 21:12:27 2023)---
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')

## ---(Fri Dec  8 07:19:16 2023)---
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')

## ---(Fri Dec  8 20:42:35 2023)---
runfile('C:/Users/hassa/.spyder-py3/temp.py', wdir='C:/Users/hassa/.spyder-py3')