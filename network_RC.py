#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:30:49 2017

@author: yty
"""
from gensim.models import KeyedVectors  
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from dataset import load_raw,form_data
" firstly, read the pre-traiend word vectors "
" and derive the sentence representation from the matrix."


" then compose the data to feed to the CNN"


" Hyper-Parameters"
num_epochs = 200
batch_size = 500
learning_rate = 0.025
dw = 50
k = 3
dwpe = 70
dc = 1000

" load data"
voc = KeyedVectors.load_word2vec_format('/home/yty/Desktop/gloveData/glove.6B.50d.txt',binary=False)
train_path = 'train.txt'
test_path = 'test.txt'
rcdataloader,test_embx,test_label = loaddata(train_path,test_path,dw,k,voc)

" define the convolutional neural network"
class Net(nn.Module):
    def __init__(self,dw,dwpe,dc,k):
        super(Net,self).__init__()
        self.l1 = nn.Linear(dw*k,dc)
        r = math.sqrt(6/(9+dc))
        self.wClass = torch.rand(9,dc)
        torch.mul(self.wClass,2*r,out = self.wClass)
        self.wClass.add(-r)
        self.l2 = nn.Linear(dc,9,bias=None)
        self.l2.weight.data = self.wClass
        
    def forward(self,x):
        x = self.l1(x)
        x = x.tanh()
        x = torch.max(x,0)[0]
        x = self.l2(x)
        return x
    
    
" define the loss and optimizer"
net = Net(dw,dwpe,dc,k) #net(Variable(train_embx[0].t()))
optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum = 0.9)
" A self-defined loss function RC_loss"
def rcloss(score,label):
    


" train the model"
for epoch in range(num_epochs):
    for i in range(len(train_label)-1):
        net.zero_grad()
        label = Variable(torch.FloatTensor([train_label[i]]))
        if label==0:
            pass
        else:
            emb = Variable(train_embx[i].t())
            output = net(emb)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
    

" test the model"    
result = []
for i in range(len(test_label)-1):
    emb = Variable(test_embx[i].t())
    real_label = test_label[i]
    output = net(emb)
    model_label = output.max(1)[1]+1
    sample = [model_label[0,0],torch.FloatTensor([real_label])]
    result.append(sample)


