import os
import numpy as np
import scipy.sparse as sp
import torch
import util

validation = False
generalized = True
preprocess = True
OVERFLOW_MARGIN = 1e-8
ADJACENCY_SCALER = 0.1
dataset = "AWA1"
npy_dir = os.path.join("/data3/huangmeixue/Dataset",dataset)
PREFINE_CLASS_path = "/data3/huangmeixue/ZSL_GCN/Model_GCN/"+dataset+"/prefine_class.pth"

class GCN_DATA(object):
    def __init__(self,dataset,preprocess,validation,generalized,preclass_path):

        data = util.DATA_LOADER(dataset,preprocess,validation,generalized)

        self.attributes_train, self.labels_train = self.get_unique_vector(data.attributes_train, data.labels_train)
        if validation:
            self.attributes_val, self.labels_val = self.get_unique_vector(data.attributes_val, data.labels_val)
        else:
            self.attributes_val, self.labels_val = self.get_unique_vector(data.attributes_test_unseen, data.labels_test_unseen)
        self.attributes_train, self.attributes_val = np.ascontiguousarray(self.attributes_train),np.ascontiguousarray(self.attributes_val)
        self.labels_train, self.labels_val = np.ascontiguousarray(self.labels_train), np.ascontiguousarray(self.labels_val)
        
        self.all_class_num = self.attributes_train.shape[0] + self.attributes_val.shape[0]
        self.all_attributes = self.get_all_attribute()
        self.adj = self.normalize_adj(self.build_adjacency(self.all_attributes))
        self.adj = self.sparse_to_tuple(self.adj)

        self.true_class_weight = self.load_class_weight(preclass_path)

    def load_class_weight(self, preclass_path):
        model = torch.load(preclass_path, map_location=lambda storage, loc: storage)
        #for name in model["state_dict"]:
        #	print(name)
        class_weight = model["state_dict"]["out.weight"].detach().cpu().numpy()
        #print(class_weight.shape)
        return class_weight

    def get_unique_vector(self, attributes, labels):
        # get unique class vector
        b = np.ascontiguousarray(labels).view(
            np.dtype((np.void, labels.dtype.itemsize * labels.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_labels = np.where(labels[idx].astype(int)==1)[1]
        unique_labels = np.flip(unique_labels, 0)
        unique_attributes = np.flip(attributes[idx], 0)
        return unique_attributes,unique_labels

    def get_all_attribute(self):
        all_attribute = np.zeros((self.all_class_num,self.attributes_train.shape[1]))
        all_attribute[self.labels_train] = self.attributes_train
        all_attribute[self.labels_val] = self.attributes_val
        return all_attribute

    def build_adjacency(self, attributes):
        # build adjacency matrix according to Euclidean distance
        squared_sum = np.sum(np.square(attributes), axis = 1)
        distances = squared_sum - 2 * np.dot(attributes, attributes.T) + np.transpose(squared_sum)
        adjacency = np.exp(-1 * distances / ADJACENCY_SCALER)
        return adjacency

    def normalize_adj(self, adj):
        # adjacency matrix is self-connect and normalize Symmetrically
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def sparse_to_tuple(self, sparse_mx):
        # Convert a scipy sparse matrix to a torch sparse tensor (tuple representation)
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

#gcn_data = GCN_DATA(dataset,preprocess,validation,generalized,PREFINE_CLASS_path)
#print(gcn_data.adj)