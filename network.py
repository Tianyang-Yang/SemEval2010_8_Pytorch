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
from dataset import loaddata
" firstly, read the pre-traiend word vectors "
" and derive the sentence representation from the matrix."


" then compose the data to feed to the CNN"


" Hyper-Parameters"
num_epochs = 200
batch_size = 10
learning_rate = 0.025
dw = 100
k = 3
dwpe = 70
dc = 1000

" load data"
voc = KeyedVectors.load_word2vec_format('/home/yty/Desktop/gloveData/glove.6B.100d.txt',binary=False)
train_path = 'train.txt'
test_path = 'test.txt'
train_loader,test_loader = loaddata(train_path,test_path,dw,k,voc,batch_size)

" define the convolutional neural network"
class Net(nn.Module):
    def __init__(self,dw,dwpe,dc,k):
        super(Net,self).__init__()
        self.l1 = nn.Linear(dw*k,dc)
        r = math.sqrt(6/(10+dc))
        self.wClass = torch.rand(10,dc)
        torch.mul(self.wClass,2*r,out = self.wClass)
        self.wClass.add(-r)
        self.l2 = nn.Linear(dc,10,bias=None)
        self.l2.weight.data = self.wClass
        
    def forward(self,x):
        x = self.l1(x)
        x = x.tanh()
        x = torch.max(x,0)[0]
        x = self.l2(x)
        return F.log_softmax(x)

" create a neural network "
net = Net(dw,dwpe,dc,k) #net(Variable(train_embx[0].t()))    

" define the loss and optimizer"
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum = 0.9)


" train the model"
" 'enumerate' is not apt for this dataloader due to the inconsisent size of the input "
" Demanding manually computing backward and optimizing step. "


for epoch in range(num_epochs):
    for i,samples in enumerate(train_loader):
        #convert tensor to Variable
        sentences = Variable(samples['sentence'])
        labels = Variable(samples['label'])
        
        #Forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(sentences)
        loss = criterion(outputs,labels)
        loss.backwward
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_loader)//batch_size, loss.data[0]))
        
#==============================================================================
#     for i in range(len(train_label)-1):
#         net.zero_grad()
#         label = Variable(torch.LongTensor([train_label[i]]))
#         if label==0:
#             pass
#         else:
#             emb = Variable(train_embx[i].t())
#             output = net(emb)
#             loss = criterion(output,label)
#             loss.backward()
#             optimizer.step()
#     
#==============================================================================

" test the model "    
correct = 0
total = 0
result = []

for sample in test_loader:
    sentence = Variable(sample['sentence'])
    outputs = net(sentence)
    _, predicted = torch.max(outputs.data,1)
    total += sample['label'].size(0)
    correct += (predicted == sample['label']).sum()

print('Accuracy of the network on the test sentences: %d %%' % (100 * correct / total))

#save the model 
torch.save(net.state_dict(), 'model.RCsoftmax')





















