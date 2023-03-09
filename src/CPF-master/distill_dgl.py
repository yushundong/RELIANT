from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp
import copy
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F

from data.utils import matrix_pow, row_normalize

from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.APPNP import APPNP
from models.PLP import PLP
from models.MLP import MLP

from utils.metrics import accuracy, my_loss


def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--assistant', type=int, default=-1, help='Different assistant teacher. -1. None 0. nasty 1. reborn')
    parser.add_argument('--student', type=str, default='PLP', help='Student Model')
    parser.add_argument('--device', type=int, default=6, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20, help='label rate')

    # for plp hyper parameters finding
    parser.add_argument('--num_layers', type=int, default=10, help='Num Layers')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedded dim for attention')
    parser.add_argument('--feat_drop', type=float, default=0.6, help='feat_dropout')
    parser.add_argument('--attn_drop', type=float, default=0.6, help='attn_dropout')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    # parser.add_argument('--temp', type=float, default=1.0, help='Temp for distilling')
    parser.add_argument('--ptype', type=int, default=0, help='Different plp architectures.'
                                                             '0. induc 1. onehot(trans) 2. only')
    parser.add_argument('--mlp_layers', type=int, default=1, help='Add feature mlp/lr')

    # for output
    parser.add_argument('--grad', type=int, default=0, help='Output Feature grad')
    parser.add_argument('--att', action='store_false', default=True, help='Output attention or not')
    parser.add_argument('--layer_flag', action='store_true', default=False, help='Layer output or not')

    return parser.parse_args()


def choose_model(conf, G, features, labels, byte_idx_train, labels_one_hot):
    if conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=conf['hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=1,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])
    elif conf['model_name'] == 'GAT':
        num_heads = 8
        num_layers = 1
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=G.ndata['feat'].shape[1],
                    num_hidden=8,
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,     # negative slope of leaky relu
                    residual=False).to(conf['device'])
    elif conf['model_name'] == 'PLP':
        model = PLP(g=G,
                    num_layers=conf['num_layers'],
                    in_dim=features.shape[1],
                    emb_dim=conf['emb_dim'],
                    num_classes=labels.max().item() + 1,
                    activation=F.relu,
                    feat_drop=conf['feat_drop'],
                    attn_drop=conf['attn_drop'],
                    residual=False,
                    byte_idx_train=byte_idx_train,
                    labels_one_hot=labels_one_hot,
                    ptype=conf['ptype'],
                    mlp_layers=conf['mlp_layers']).to(conf['device'])
    elif conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=G.ndata['feat'].shape[1],
                          n_hidden=16,
                          n_classes=labels.max().item() + 1,
                          n_layers=1,
                          activation=F.relu,
                          dropout=0.5,
                          aggregator_type=conf['agg_type']).to(conf['device'])
    elif conf['model_name'] == 'APPNP':
        model = APPNP(g=G,
                      in_feats=G.ndata['feat'].shape[1],
                      hiddens=[64],
                      n_classes=labels.max().item() + 1,
                      activation=F.relu,
                      feat_drop=0.5,
                      edge_drop=0.5,
                      alpha=0.1,
                      k=10).to(conf['device'])
    elif conf['model_name'] == 'LogReg':
        model = MLP(num_layers=1,
                    input_dim=G.ndata['feat'].shape[1],
                    hidden_dim=None,
                    output_dim=labels.max().item() + 1,
                    dropout=0).to(conf['device'])
    elif conf['model_name'] == 'MLP':
        model = MLP(num_layers=2,
                    input_dim=G.ndata['feat'].shape[1],
                    hidden_dim=conf['hidden'],
                    output_dim=labels.max().item() + 1,
                    dropout=conf['dropout']).to(conf['device'])
    else:
        raise ValueError(f'Undefined Model.')
    return model


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


def approx_func(s):
    x = 2 * s - 1
    return 1 / 2 + 1 / 2 * x - 1 / 8 / 2 * (5 * x * x * x - 3 * x) + 1 / 16 / 8 * (
                63 * x * x * x * x * x - 70 * x * x * x + 15 * x) - 5 / 128 / 16 * (
                       429 * x * x * x * x * x * x * x - 693 * x * x * x * x * x + 315 * x * x * x - 35 * x) + 7 / 256 / 128 * (
                       12155 * x * x * x * x * x * x * x * x * x - 25740 * x * x * x * x * x * x * x + 18018 * x * x * x * x * x - 4620 * x * x * x + 315 * x)


def approx_loss(logits, sens, idx):
    g1 = np.argwhere(sens == 0.0).reshape(-1)
    g2 = np.argwhere(sens == 1.0).reshape(-1)
    idx_set = set(idx.cpu().numpy())
    g1 = np.array(list(set(g1) & idx_set))
    g2 = np.array(list(set(g2) & idx_set))
    loss = torch.square(approx_func(logits[g1]).sum(axis=0) / g1.shape[0] - approx_func(logits[g2]).sum(axis=0) / g2.shape[0]).sum()
    return loss


def approx_loss_eo(logits, sens, labels, idx):
    g1 = np.argwhere(sens == 0).reshape(-1)
    g2 = np.argwhere(sens == 1).reshape(-1)
    g = np.argwhere(labels.cpu().numpy() == 1).reshape(-1)
    idx_set = set(idx.cpu().numpy())
    g1 = np.array(list(set(g1) & set(g) & idx_set))
    g2 = np.array(list(set(g2) & set(g) & idx_set))
    loss = torch.square(approx_func(logits[g1]).sum(axis=0) / g1.shape[0] - approx_func(logits[g2]).sum(axis=0) / g2.shape[0]).sum()
    return loss


def distill_train(all_logits, dur, epoch, model, proxy, features, optimizer, optimizer_proxy, conf, G, sens, labels_init, labels, idx_no_train, idx_train,
                  idx_val, idx_test, cas):
    t0 = time.time()
    model.train()
    torch.autograd.set_detect_anomaly(True)
    if proxy is not None:
        feature = torch.concat([features, proxy.mean(axis=0).repeat([features.shape[0], 1])], 1)
    else:
        feature = features
    if conf['model_name'] in ['GCN', 'APPNP', 'LogReg', 'MLP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        logits = model(G.ndata['feat'])[0]
    elif conf['model_name'] == 'GraphSAGE':
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'PLP':
        logits = model(feature, labels_init)[0]
    logp = F.log_softmax(logits, dim=1)
    # we only compute loss for labeled nodes
    if conf['model_name'] == 'PLP':
        if proxy is not None:
            if conf['metric'] == 'sp':
                loss1 = my_loss(logp[idx_no_train], cas[-1][idx_no_train]) + conf['alpha'] * approx_loss(torch.exp(logp), sens, idx_no_train)
            elif conf['metric'] == 'eo':
                loss1 = my_loss(logp[idx_no_train], cas[-1][idx_no_train]) + conf['alpha'] * approx_loss_eo(torch.exp(logp), sens, labels, idx_no_train)
            else:
                raise
        else:
            loss1 = my_loss(logp[idx_no_train], cas[-1][idx_no_train])
    else:
        loss1 = F.kl_div(logp, cas[-1], reduction='batchmean')
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    acc_train = accuracy(logp[idx_train], labels[idx_train])

    if optimizer_proxy is not None:
        feature = torch.concat([features, proxy], 1)
        logits = model(feature, labels_init)[0]
        logp = F.log_softmax(logits, dim=1)
        optimizer_proxy.zero_grad()
        if conf['metric'] == 'sp':
            loss2 = -conf['alpha'] * approx_loss(torch.exp(logp), sens, idx_no_train)
        elif conf['metric'] == 'eo':
            loss2 = -conf['alpha'] * approx_loss_eo(torch.exp(logp), sens, labels, idx_no_train)
        else:
            raise
        loss2.backward()
        optimizer_proxy.step()
    else:
        loss2 = torch.zeros(1)
    dur.append(time.time() - t0)
    model.eval()
    if proxy is not None:
        feature = torch.concat([features, proxy.mean(axis=0).repeat([features.shape[0], 1])], 1)
    else:
        feature = features
    if conf['model_name'] in ['GCN', 'APPNP', 'LogReg', 'MLP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        logits = model(G.ndata['feat'])[0]
    elif conf['model_name'] == 'GraphSAGE':
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'PLP':
        logits = model(feature, labels_init)[0]
    logp = F.log_softmax(logits, dim=1)
    all_logits.append(logp.cpu().detach().numpy())
    loss_val = my_loss(logp[idx_val], cas[-1][idx_val])
    acc_val = accuracy(logp[idx_val], labels[idx_val])
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    preds = logp.max(1)[1].type_as(labels).cpu().numpy()
    parity, equality = fair_metric(preds[idx_test.cpu().numpy()], labels.cpu().numpy()[idx_test.cpu().numpy()], sens[idx_test.cpu().numpy()])
    # print(
    #     'Epoch %d | Loss1: %.4f | Loss2: %.4f | loss_val: %.4f | acc_train: %.4f | acc_val: %.4f | acc_test: %.4f | parity: %.4f | equality: %.4f | Time(s) %.4f' % (
    #     epoch, loss1.item(), loss2.item(), loss_val.item(), acc_train.item(), acc_val.item(), acc_test.item(), parity, equality, dur[-1]))

    return acc_val, loss_val


def model_train(conf, model, proxy, features, optimizer, optimizer_proxy, G, sens, labels_init,
                labels, idx_no_train, idx_train, idx_val, idx_test, cas):
    all_logits = []
    dur = []
    best = 0
    cnt = 0
    for epoch in tqdm(range(conf['max_epoch'])):
        acc_val, loss_val = distill_train(all_logits, dur, epoch, model, proxy, features, optimizer, optimizer_proxy, conf, G, sens, labels_init, labels,
                                          idx_no_train, idx_train, idx_val, idx_test, cas)
        epoch += 1
        if acc_val >= best:
            best = acc_val
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            cnt = 0
        else:
            cnt += 1
        if cnt == conf['patience'] or epoch == conf['max_epoch']:
            print("Stop!!!")
            break
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(np.sum(dur)))
    return best


def distill_test(conf, model, G, labels_init, labels, idx_test, cas):
    model.eval()

    if conf['model_name'] in ['GCN', 'APPNP', 'LogReg', 'MLP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        logits, G.edata['a'] = model(G.ndata['feat'])
    elif conf['model_name'] == 'GraphSAGE':
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'PLP':
        logits = model(G.ndata['feat'], labels_init)[0]
    logp = F.log_softmax(logits, dim=1)
    loss_test = F.nll_loss(logp[idx_test], labels[idx_test])
    preds = torch.argmax(logp, dim=1).cpu().detach()
    teacher_preds = torch.argmax(cas[-1], dim=1).cpu().detach()
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    acc_teacher_test = accuracy(cas[-1][idx_test], labels[idx_test])
    same_predict = np.count_nonzero(teacher_preds[idx_test] == preds[idx_test]) / len(idx_test)
    acc_dis = np.abs(acc_teacher_test.item() - acc_test.item())
    # print("Test set results: loss= {:.4f} acc_test= {:.4f} acc_teacher_test= {:.4f} acc_dis={:.4f} same_predict= {:.4f}".format(
    #     loss_test.item(), acc_test.item(), acc_teacher_test.item(), acc_dis, same_predict))

    return loss_test, acc_test, logp, same_predict


def save_output(output_dir, preds, labels, output, acc_test, same_predict, G, idx_train, adj, conf):
    acc_test = acc_test.cpu().numpy()
    np.savetxt(output_dir.joinpath('preds.txt'), preds, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('labels.txt'), labels, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('output.txt'), output, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('test_acc.txt'), np.array([acc_test]), fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('same_predict.txt'), np.array([same_predict]), fmt='%.4f', delimiter='\t')
    if 'a' in G.edata:
        edge = torch.stack((G.edges()[0], G.edges()[1]), 0)
        sp_att = sp.coo_matrix((G.edata['a'].cpu().detach().numpy(), edge.cpu()), shape=adj.cpu().size())
        sp.save_npz(output_dir.joinpath('attention_weight.npz'), sp_att, compressed=True)
        att_torch = torch.FloatTensor(sp_att.todense())
        att_torch[idx_train, :] = torch.eye(len(adj))[idx_train, :]
        # k_propagate_prob = quick_matrix_pow(propagate_prob, epoch)
        k_propagate_prob = matrix_pow(att_torch, conf['num_layers'], att_torch[:, idx_train])
        sp_k_att = sp.coo_matrix(row_normalize(k_propagate_prob))
        sp.save_npz(output_dir.joinpath('k_attention_weight.npz'), sp_k_att, compressed=True)
    if 'alpha' in G.ndata:
        lr_ratio = G.ndata['alpha'].cpu().detach().numpy()
        el = G.ndata['el'].cpu().detach().numpy()
        er = G.ndata['er'].cpu().detach().numpy()
        np.savetxt(output_dir.joinpath('lr_ratio.txt'), lr_ratio, fmt='%.4f', delimiter='\t')
        np.savetxt(output_dir.joinpath('el.txt'), el, fmt='%.4f', delimiter='\t')
        np.savetxt(output_dir.joinpath('er.txt'), er, fmt='%.4f', delimiter='\t')
    if conf['grad'] == 1:
        grad = G.ndata['feat'].grad.cpu().numpy()
        np.savetxt(output_dir.joinpath('grad.txt'), grad, fmt='%.4f', delimiter='\t')
    if conf['model_name'] == 'PLP':
        with open(output_dir.joinpath('sta_log'), 'a') as hyper:
            hyper.write(str(conf['num_layers']) + '\t')
            hyper.write(str(conf['mlp_layers']) + '\t')
            hyper.write(str(conf['emb_dim']) + '\t')
            hyper.write(str(conf['feat_drop']) + '\t')
            hyper.write(str(conf['attn_drop']) + '\t')
            hyper.write(str(conf['lr']) + '\t')
            hyper.write(str(conf['wd']) + '\t')
            hyper.write('%.4f\n' % acc_test)
