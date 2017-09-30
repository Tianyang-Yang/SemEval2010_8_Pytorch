#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:53:43 2017

@author: yty
"""
from gensim.models import KeyedVectors  
import torch
import numpy as np
import re

def read_all_lines(path):
    all_lines = []
    with open(path, 'r', encoding='utf-8') as file:
        temp_lines = file.readlines()
        for line in temp_lines:
            line = line.strip()
            if line:
                all_lines.append(line)
    return all_lines


" To load the raw data to be haddled"
def load_raw(path1,path2):
    al_train = read_all_lines(path1)
    al_test = read_all_lines(path2)
    train_raw = []
    test_raw = []
    i=0
    for line in al_train:
        train_raw.append(line.split('|'))
        i = i+1
    i=0
    for line in al_test:
        test_raw.append(line.split('|'))
        i= i+1
    return train_raw,test_raw

"convert the character label to the index(int)"
def label2index(label):
    if label=='Other':
        return 0
    if label=='Cause-Effect':
        return 1
    if label=='Component-Whole':
        return 2
    if label=='Content-Container':
        return 3
    if label=='Entity-Destination':
        return 4
    if label=='Entity-Origin':
        return 5
    if label=='Instrument-Agency':
        return 6
    if label=='Member-Collection':
        return 7
    if label=='Message-Topic':
        return 8
    else: "Product-Producer"
        return 9

" generate a integrate number sequence
def generate_range(start,stop):
    result = []
    for i in range(start,stop):
        result.append(i)
    return result


" form the data that can be fed to the neural networks"
def form_data(raw1,raw2,dw,voc):
    """
    @parameter:
        raw1,raw2:  the data from Load_raw
        dw :  the length of embedding
        voc:  the word2vec vocabulary
    @return:
        matrix emb and label of train and test dataset, respectively
    """
    train_embx = []
    train_label = []
    test_embx = []
    test_label = []
    for sample in raw1:
        train_label.append(label2index(sample[0]))
        sentence = sample[3].split(' ')
        n = len(sentence)
        sentence.append(sentence[len(sentence)-1])
        sentence.insert(0,sentence[0])
        zemb = torch.zeros(k*dw,1)
        for i in range(n-1):
            window = sentence[slice(i,i+3)]
            z = torch.zeros(1,1)
            for word in window:
                if word in voc.vocab:
                    neww = torch.from_numpy(np.array(voc.word_vec(word)))
                else:
                    neww = torch.zeros(dw,1)
                z = torch.cat((z,neww),0)
            indices = generate_range(1,k*dw+1)
            z = torch.index_select(z,0,torch.LongTensor(indices))
            zemb = torch.cat((zemb,z),1)
        indices = generate_range(int(sample[1])-1,int(sample[2]))
        zemb = torch.index_select(zemb,1,torch.LongTensor(indices))
        train_embx.append(zemb)
    
    for sample in raw2:
        test_label.append(label2index(sample[0]))
        sentence = sample[3].split(' ')
        n = len(sentence)
        sentence.append(sentence[len(sentence)-1])
        sentence.insert(0,sentence[0])
        zemb = torch.zeros(k*dw,1)
        for i in range(n-1):
            window = sentence[slice(i,i+3)]
            z = torch.zeros(1,1)
            for word in window:
                if word in voc.vocab:
                    neww = torch.from_numpy(np.array(voc.word_vec(word)))
                else:
                    neww = torch.zeros(dw,1)
                z = torch.cat((z,neww),0)
            indices = generate_range(1,k*dw+1)
            z = torch.index_select(z,0,torch.LongTensor(indices))
            zemb = torch.cat((zemb,z),1)
        indices = generate_range(int(sample[1])-1,int(sample[2]))
        zemb = torch.index_select(zemb,1,torch.LongTensor(indices))
        test_embx.append(zemb)
    return train_embx,train_label,test_embx,test_label
#train_embx,train_label,test_embx,test_label = form_data(train_raw,test_raw,50)

if __name__ == '__main__':
    train_path = 'train.txt'
    test_path = 'test.txt'
    voc = KeyedVectors.load_word2vec_format('/home/yty/Desktop/gloveData/glove.6B.50d.txt',binary=False)
    train_raw,test_raw = load_raw(train_path,test_path)
    k = 3
    dw = 50
    train_embx,train_label,test_embx,test_label = form_data(train_raw,test_raw,50,voc)