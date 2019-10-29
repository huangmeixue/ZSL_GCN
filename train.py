from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from convert_to_graph import GCN_DATA
from model import GCN

EPOCH = 1000
weight_decay = 0.0005
LR = 0.001
hidden_size = 1024
dropout = 0.5
normalize = True

validation = False
generalized = True
preprocess = True
OVERFLOW_MARGIN = 1e-8
ADJACENCY_SCALER = 0.1
dataset = "AWA1"
PREFINE_CLASS_path = "/data3/huangmeixue/ZSL_GCN/Model_GCN/"+dataset+"/prefine_class.pth"

seed = 9182
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# load data
gcn_data = GCN_DATA(dataset,preprocess,validation,generalized,PREFINE_CLASS_path)

# Model and optimizer
model = GCN(in_feature=gcn_data.all_attributes.shape[1],
            hidden_size=hidden_size,
            out_feature=gcn_data.true_class_weight.shape[1],
            dropout=dropout)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
loss_fun = torch.nn.MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    loss_fun = loss_fun.cuda()

def train(gcn_data, epoch):
    if torch.cuda.is_available():
        adj = gcn_data.adj.cuda()
        true_class_weight = torch.FloatTensor(gcn_data.true_class_weight).cuda()
        all_attributes = torch.FloatTensor(gcn_data.all_attributes).cuda()
        #attributes_train = torch.FloatTensor(gcn_data.attributes_train).cuda()
        #attributes_val = torch.FloatTensor(gcn_data.attributes_val).cuda()
        labels_train = torch.LongTensor(gcn_data.labels_train).cuda()
        #labels_val = torch.LongTensor(gcn_data.labels_val).cuda()

    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    pred_class_weight = model(all_attributes, adj)
    if normalize:
    	pred_class_weight = F.normalize(pred_class_weight)
    	true_class_weight = F.normalize(true_class_weight)
    loss_train = loss_fun(pred_class_weight[labels_train], true_class_weight)
    #cls = classifier.Classifier(features_train,labels_train,data,gzsl_classifier_path,lr=0.0001,batch_size=64,epoch=30,validation=False,generalized=True)
    #acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          #'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - start_time))


for epoch in range(EPOCH):
    train(gcn_data, epoch)