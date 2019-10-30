from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

from convert_to_graph import GCN_DATA
from model import GCN
import util

EPOCH = 3000
weight_decay = 0.0005
LR = 0.001
#hiddens_dim = [2048,2048,1024,1024,512]
#dropout = [0.3,0.3,0.3,0.3,0.3,0]
hiddens_dim = [1049]
dropout = [0.3,0]
normalize = True
BATCH_SIZE = 64

validation = False
generalized = False
preprocess = True
OVERFLOW_MARGIN = 1e-8
ADJACENCY_SCALER = 0.01
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
model = GCN(input_dim=gcn_data.all_attributes.shape[1],
            hiddens_dim=hiddens_dim,
            output_dim=gcn_data.true_class_weight.shape[1],
            dropout=dropout)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
loss_fun = torch.nn.MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    loss_fun = loss_fun.cuda()

#为了分类，将one-hot表示的标签向量转化为索引表示
def map_label(labels):
    index_labels = np.where(labels.astype(int)==1)[1]
    unique_class = np.unique(index_labels)
    class_num = len(unique_class)
    #print(unique_class,len(unique_class))
    mapped_label =  np.zeros((labels.shape[0],))
    for i in range(class_num):
        mapped_label[index_labels==unique_class[i]] = i
    mapped_label = mapped_label.astype(int)
    #print("Number of Classes: {}".format(class_num))
    return mapped_label

def train(gcn_data, raw_data, epoch):
    if torch.cuda.is_available():
        adj = gcn_data.adj.cuda()
        true_class_weight = torch.FloatTensor(gcn_data.true_class_weight).cuda()
        all_attributes = torch.FloatTensor(gcn_data.all_attributes).cuda()
        #attributes_train = torch.FloatTensor(gcn_data.attributes_train).cuda()
        #attributes_val = torch.FloatTensor(gcn_data.attributes_val).cuda()
        labels_train = torch.LongTensor(gcn_data.labels_train).cuda()
        labels_val = torch.LongTensor(gcn_data.labels_val).cuda()

    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    pred_class_weight = model(all_attributes, adj)
    if normalize:
    	pred_class_weight = F.normalize(pred_class_weight)
    	true_class_weight = F.normalize(true_class_weight)
    loss_train = loss_fun(pred_class_weight[labels_train], true_class_weight)
    loss_train.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        acc_train = test(raw_data.features_train, map_label(raw_data.labels_train), pred_class_weight[labels_train])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'time: {:.4f}s'.format(time.time() - start_time))

        validate(raw_data, all_attributes, adj, true_class_weight, labels_train, labels_val)
        '''
        model.eval()
        with torch.no_grad():
            pred_class_weight = model(all_attributes, adj)
        loss_val = loss_fun(pred_class_weight[labels_train], true_class_weight)
        if validation:
            acc_val = test(raw_data.features_val, map_label(raw_data.labels_val), pred_class_weight[labels_val])
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()))
        else:
            if generalized:
                labels_test_seen = np.where(raw_data.labels_test_seen==1)[1]
                labels_test_unseen = np.where(raw_data.labels_test_unseen==1)[1]
                acc_test_seen = test(raw_data.features_test_seen, labels_test_seen, pred_class_weight)
                acc_test_unseen = test(raw_data.features_test_unseen, labels_test_unseen, pred_class_weight)
                acc_H = 2 * acc_test_seen * acc_test_unseen / (acc_test_seen + acc_test_unseen)
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_H: {:.4f}'.format(acc_H.item()),
                      'acc_test_seen: {:.4f}'.format(acc_test_seen.item()),
                      'acc_test_unseen: {:.4f}'.format(acc_test_unseen.item()))
            else:
                acc_test_unseen = test(raw_data.features_test_unseen, map_label(raw_data.labels_test_unseen), pred_class_weight[labels_val])
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_test_unseen: {:.4f}'.format(acc_test_unseen.item()))
        '''
def validate(raw_data, attributes, adj, true_class_weight, labels_train, labels_val):
    model.eval()
    with torch.no_grad():
        pred_class_weight = model(attributes, adj)
    loss_val = loss_fun(pred_class_weight[labels_train], true_class_weight)
    if validation:
        acc_val = test(raw_data.features_val, map_label(raw_data.labels_val), pred_class_weight[labels_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))
    else:
        if generalized:
            labels_test_seen = np.where(raw_data.labels_test_seen==1)[1]
            labels_test_unseen = np.where(raw_data.labels_test_unseen==1)[1]
            acc_test_seen = test(raw_data.features_test_seen, labels_test_seen, pred_class_weight)
            acc_test_unseen = test(raw_data.features_test_unseen, labels_test_unseen, pred_class_weight)
            acc_H = 2 * acc_test_seen * acc_test_unseen / (acc_test_seen + acc_test_unseen)
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_H: {:.4f}'.format(acc_H.item()),
                  'acc_test_seen: {:.4f}'.format(acc_test_seen.item()),
                  'acc_test_unseen: {:.4f}'.format(acc_test_unseen.item()))
        else:
            acc_test_unseen = test(raw_data.features_test_unseen, map_label(raw_data.labels_test_unseen), pred_class_weight[labels_val])
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_test_unseen: {:.4f}'.format(acc_test_unseen.item()))

def accuracy(scores, true_label, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = true_label.size(0)

        _, pred_label = scores.topk(maxk, 1, True, True)
        pred_label = pred_label.t()
        correct = pred_label.eq(true_label.view(1, -1).expand_as(pred_label))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result

def test(features, labels, class_weight):
    topKs = [1]
    top_retrv = [1, 2, 5, 10]
    hit_count = np.zeros((len(topKs), len(top_retrv)), dtype=np.float32)
    cnt_valid = 0
    start_time = time.time()

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    dataset = Data.TensorDataset(features,labels)
    loader = Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=False)
    
    class_weight = class_weight.permute(1,0)

    for i, (feature, label) in enumerate(loader):
        if torch.cuda.is_available():
            feature, label = feature.cuda(), label.cuda()

        cnt_valid += feature.size(0)
        
        scores = torch.matmul(feature, class_weight).squeeze()
        batch_result = accuracy(scores, label, top_retrv)

        for k in range(len(topKs)):
            for k2 in range(len(top_retrv)):
                hit_count[k][k2] = hit_count[k][k2] + float(batch_result[k2]/100*feature.size(0))
        
        #if cnt_valid % 1 == 0:
        #    inter = time.time() - start_time
        #    print('Processing %d / %d ' % (cnt_valid, len(dataset)), ', Remaining Estimated Time: ', inter / (i+1) * (len(loader) - i - 1))
        #    print(hit_count / cnt_valid)

    topk_result = hit_count / cnt_valid
    return topk_result[0][0]

raw_data = util.DATA_LOADER(dataset,preprocess,validation,generalized)
for epoch in range(EPOCH):
    train(gcn_data, raw_data, epoch)