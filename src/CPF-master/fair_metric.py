from data.utils import load_tensor_data
from distill_dgl import fair_metric
import torch
import numpy as np


dataset = 'german'
path = './outputs/' + dataset + '/GCN/cascade_random_0_20/'
adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test, sens = load_tensor_data(None, dataset, 20, torch.device('cpu'))

preds = np.loadtxt(path+'preds.txt')
labels = np.loadtxt(path+'labels.txt')
parity, equality = fair_metric(preds[idx_test], labels[idx_test], sens[idx_test])
print([parity, equality, (preds==labels).sum() / labels.shape[0]])
# np.save('../pytorch-gnn-meta-attack/data/'+dataset+'/adj.npy', adj.numpy())
# np.save('../pytorch-gnn-meta-attack/data/'+dataset+'/feature.npy', features.numpy())
# np.save('../pytorch-gnn-meta-attack/data/'+dataset+'/label.npy', labels.numpy())
# np.save('../pytorch-gnn-meta-attack/data/'+dataset+'/train_set.npy', idx_train.numpy())
# np.save('../pytorch-gnn-meta-attack/data/'+dataset+'/val_set.npy', idx_val.numpy())
# np.save('../pytorch-gnn-meta-attack/data/'+dataset+'/test_set.npy', idx_test.numpy())
# np.save('../pytorch-gnn-meta-attack/data/'+dataset+'/sens.npy', sens)
