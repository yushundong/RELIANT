import numpy as np
import scipy.sparse as sp
import torch
import os
import time
import pandas as pd
from pathlib import Path
from scipy.spatial import distance_matrix
from data.get_dataset import load_dataset_and_split


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj


def normalize_features(features):
    features = normalize(features)
    return features


def initialize_label(idx_train, labels_one_hot):
    labels_init = torch.ones_like(labels_one_hot) / len(labels_one_hot[0])
    #labels_init = torch.ones_like(labels_one_hot) / 3
    #print(labels_init)
    labels_init[idx_train] = labels_one_hot[idx_train]
    return labels_init


def split_double_test(dataset, idx_test):
    test_num = len(idx_test)
    idx_test1 = idx_test[:int(test_num/2)]
    idx_test2 = idx_test[int(test_num/2):]
    return idx_test1, idx_test2


def preprocess_adj(model_name, adj):
    return normalize_adj(adj)


def preprocess_features(model_name, features):
    return features


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map


def load_ogb_data(dataset, device):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(name="ogbn-"+dataset, root='data')
    splitted_idx = data.get_idx_split()
    idx_train, idx_val, idx_test = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    labels = labels.squeeze()
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    graph = graph.remove_self_loop().add_self_loop()
    features = graph.ndata['feat']
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    return graph, features, labels, idx_train, idx_val, idx_test


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="../../../../data/raw_materials/bail/"):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    if os.path.exists(os.path.join(path, 'split')):
        idx_train = np.load(os.path.join(path, 'split', 'train_split.npy'))
        idx_val = np.load(os.path.join(path, 'split', 'val_split.npy'))
        idx_test = np.load(os.path.join(path, 'split', 'test_split.npy'))
    else:
        import random
        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:int(0.3 * len(label_idx_0))],
                              label_idx_1[:int(0.3 * len(label_idx_1))])
        idx_val = np.append(label_idx_0[int(0.3 * len(label_idx_0)):int(0.5 * len(label_idx_0))],
                            label_idx_1[int(0.3 * len(label_idx_1)):int(0.5 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.5 * len(label_idx_0)):], label_idx_1[int(0.5 * len(label_idx_1)):])
        os.mkdir(os.path.join(path, 'split'))
        np.save(os.path.join(path, 'split', 'train_split.npy'), idx_train)
        np.save(os.path.join(path, 'split', 'val_split.npy'), idx_val)
        np.save(os.path.join(path, 'split', 'test_split.npy'), idx_test)

    sens = idx_features_labels[sens_attr].values.astype(int)
    labels = np.vstack([1 - labels, labels]).T

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="../../../../data/raw_materials/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    # Normalize LoanAmount
    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
    
    # Normalize Age
    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
    
    # Normalize LoanDuration
    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
    #
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    import random
    random.seed(0)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)

    labels = np.vstack([1 - labels, labels]).T

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_aminer(dataset, path="../../../../data/raw_materials/"):

    X = np.load(os.path.join(path, dataset, 'X.npz'))
    features = sp.csr_matrix((X['data'], (X['row'], X['col'])), shape=X['shape'], dtype=np.float32)
    labels = np.loadtxt(os.path.join(path, dataset, 'labels.txt'), dtype=np.float32)[:, 1]
    sens = np.loadtxt(os.path.join(path, dataset, 'sens.txt'), dtype=np.float32)[:, 1]
    edges = np.loadtxt(os.path.join(path, dataset, 'edgelist.txt'), dtype=int)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    
    sens_ind = sens != 2
    sens[sens_ind] = 0
    sens[~sens_ind] = 1
    labels_ind = labels != 3
    labels[labels_ind] = 0
    labels[~labels_ind] = 1

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    if os.path.exists(os.path.join(path, dataset, 'split')):
        idx_train = np.load(os.path.join(path, dataset, 'split', 'train_split.npy'))
        idx_val = np.load(os.path.join(path, dataset, 'split', 'val_split.npy'))
        idx_test = np.load(os.path.join(path, dataset, 'split', 'test_split.npy'))
    else:
        import random
        random.seed(20)
        label_idx = []
        for i in range(int(labels.min()), int(labels.max()) + 1):
            label_idx.append(np.where(labels == i)[0])
            random.shuffle(label_idx[-1])
        idx_train = np.array([])
        idx_val = np.array([])
        idx_test = np.array([])
        for i in range(len(label_idx)):
            idx_train = np.append(idx_train, label_idx[i][:int(0.5 * len(label_idx[i]))])
            idx_val = np.append(idx_val, label_idx[i][int(0.5 * len(label_idx[i])):int(0.75 * len(label_idx[i]))])
            idx_test = np.append(idx_test, label_idx[i][int(0.75 * len(label_idx[i])):])
        os.mkdir(os.path.join(path, dataset, 'split'))
        np.save(os.path.join(path, dataset, 'split', 'train_split.npy'), idx_train)
        np.save(os.path.join(path, dataset, 'split', 'val_split.npy'), idx_val)
        np.save(os.path.join(path, dataset, 'split', 'test_split.npy'), idx_test)

    # labels = np.vstack([labels, 1 - labels]).T
    labels = np.eye(int(labels.max()) + 1)[labels.astype(int)]

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_tensor_data(model_name, dataset, labelrate, device):
    if dataset in ['composite', 'composite2', 'composite3']:
        adj, features, labels_one_hot, idx_train, idx_val, idx_test = load_composite_data(dataset)
    elif dataset == 'german':
        adj, features, labels_one_hot, idx_train, idx_val, idx_test, sens = load_german(dataset)
    elif dataset == 'bail':
        adj, features, labels_one_hot, idx_train, idx_val, idx_test, sens = load_bail(dataset)
    elif dataset in ['small', 'medium']:
        adj, features, labels_one_hot, idx_train, idx_val, idx_test, sens = load_aminer(dataset)
    else:
        adj, features, labels_one_hot, idx_train, idx_val, idx_test, sens = load_dataset_and_split(labelrate, dataset)
    adj = preprocess_adj(model_name, adj)
    features = preprocess_features(model_name, features)
    adj_sp = adj.tocoo()
    values = torch.FloatTensor(adj_sp.data)
    indices = torch.LongTensor(np.vstack((adj_sp.row, adj_sp.col)))
    shape = adj.shape
    adj = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = labels_one_hot.argmax(axis=1)
    labels = torch.LongTensor(labels)
    labels_one_hot = torch.FloatTensor(labels_one_hot)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print('Device: ', device)
    features = features.to(device)
    labels = labels.to(device)
    labels_one_hot = labels_one_hot.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    if dataset in ['bail','credit', 'raw', 'german', 'medium', 'small']:
        return adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test, sens
    else:
        return adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test


def load_composite_data(dataset):
    base_dir = Path.cwd().joinpath('data', dataset)
    adj = np.loadtxt(str(base_dir.joinpath('adj')))
    features = np.loadtxt(str(base_dir.joinpath('features')))
    labels_one_hot = np.loadtxt(str(base_dir.joinpath('labels')))
    idx_train = np.loadtxt(str(base_dir.joinpath('idx_train')))
    idx_val = np.loadtxt(str(base_dir.joinpath('idx_val')))
    idx_test = np.loadtxt(str(base_dir.joinpath('idx_test')))
    adj = sp.csr_matrix(adj)
    # adj = normalize_adj(adj)
    features = sp.csr_matrix(features)
    # features = normalize_features(features)
    # labels, labels_init = initialize_label(idx_train, labels_one_hot)

    return adj, features, labels_one_hot, idx_train, idx_val, idx_test


def table_to_dict(adj):
    adj = adj.cpu().numpy()
    # print(adj)
    # adj = adj.todense()
    adj_list = dict()
    for i in range(len(adj)):
        adj_list[i] = set(np.argwhere(adj[i] > 0).ravel())
    return adj_list


def matrix_pow(m1, n, m2):
    t = time.time()
    m1 = sp.csr_matrix(m1)
    m2 = sp.csr_matrix(m2)
    ans = m1.dot(m2)
    for i in range(n-2):
        ans = m1.dot(ans)
    ans = torch.FloatTensor(ans.todense())
    print(time.time() - t)
    return ans


def quick_matrix_pow(m, n):
    t = time.time()
    E = torch.eye(len(m))
    while n:
        if n % 2 != 0:
            E = torch.matmul(E, m)
        m = torch.matmul(m, m)
        n >>= 1
    print(time.time() - t)
    return E


def row_normalize(data):
    return (data.t() / torch.sum(data.t(), dim=0)).t()


def np_normalize(matrix):
    from sklearn.preprocessing import normalize
    """Normalize the matrix so that the rows sum up to 1."""
    matrix[np.isnan(matrix)] = 0
    return normalize(matrix, norm='l1', axis=1)


def check_writable(dir, overwrite=False):
    import shutil
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif overwrite:
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        pass


def check_readable(dir):
    if not os.path.exists(dir):
        print(dir)
        raise ValueError(f'No such a directory or file!')


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


