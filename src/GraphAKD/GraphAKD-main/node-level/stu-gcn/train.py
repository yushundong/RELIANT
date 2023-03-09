import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from logger import output_results
from collections import defaultdict, namedtuple
from gcn import GCN, SGC
import random
import os
import dgl.function as fn
import scipy.sparse as sp
import itertools
from data.utils import load_tensor_data

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.random.seed(seed)


def compute_micro_f1(logits, y, mask=None):
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class logits_D(nn.Module):
    def __init__(self, n_class, n_hidden):
        super(logits_D, self).__init__()
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.lin = nn.Linear(self.n_hidden, self.n_hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.n_hidden, self.n_class+1, bias=False)

    def forward(self, logits, temperature=1.0):
        out = self.lin(logits / temperature)
        out = logits + out
        out = self.relu(out)
        dist = self.lin2(out)
        return dist


class local_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(local_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden, )))
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 0.5))

    def forward(self, emb, g):
        emb = F.normalize(emb, p=2)
        g.ndata['e'] = emb
        g.ndata['ew'] = emb @ torch.diag(self.d)
        g.apply_edges(fn.u_dot_v('ew', 'e', 'z'))
        pair_dis = g.edata['z']
        return pair_dis * self.scale

class global_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(global_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden, )))
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 0.5))

    def forward(self, emb, summary):
        emb = F.normalize(emb, p=2)
        sim = emb @ torch.diag(self.d)
        assert summary.shape[-1] == 1
        sim = sim @ summary
        return sim * self.scale


def approx_func(s):
    x = 2 * s - 1
    # return 1/2 + 1/2*x - 1/8/2*(5*x*x*x-3*x) + 1/16/8*(63*x*x*x*x*x-70*x*x*x+15*x)
    return 1/2 + 1/2*x - 1/16*(5*x*x*x-3*x) + 1/16/8*(63*x*x*x*x*x-70*x*x*x+15*x) - 5/128/16*(429*x*x*x*x*x*x*x-693*x*x*x*x*x+315*x*x*x-35*x) + 7/256/128*(
           12155*x*x*x*x*x*x*x*x*x-25740*x*x*x*x*x*x*x+18018*x*x*x*x*x-4620*x*x*x+315*x)


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


def run(args, g, n_classes, cuda, n_running):
    CUDA_LAUNCH_BLOCKING=1
    set_random_seed(args)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    if args.proxy in [1, 2]:
        in_feats += 2
    n_edges = g.number_of_edges()

    # load teacher knowledge
    if args.role == 'stu':
        kd_dir = '../../distilled'
        if args.dataset in ['bail', 'credit', 'small', 'medium']:
            kd_path = os.path.join(kd_dir, args.dataset + '_' + args.teacher + '.pth.tar')
        else:
            kd_path = os.path.join(kd_dir, args.dataset + f'-knowledge.pth.tar')
            assert os.path.isfile(kd_path), "Please download teacher knowledge first"
        knowledge = torch.load(kd_path, map_location=g.device)
        tea_logits = knowledge['logits']
        tea_emb = knowledge['embedding']  # torch.Size([2708, 64])
        tea_parity = knowledge['parity']
        tea_equality = knowledge['equality']
        if 'perm' in knowledge.keys() and args.dataset in ['arxiv', 'reddit']:
            perm = knowledge['perm']
            inv_perm = perm.sort()[1]
            tea_logits = tea_logits[inv_perm]
            tea_emb = tea_emb[inv_perm]
        tea_acc = compute_micro_f1(tea_logits, labels, test_mask)  # for val
        print(f'Teacher Test SCORE: {tea_acc:.3%}')
        print(f'Teacher Parity SCORE: {tea_parity}')
        print(f'Teacher Equality SCORE: {tea_equality}')

    # create student proxy
    _, _, _, _, _, idx_train, idx_val, idx_test, sens = load_tensor_data(args.teacher, args.dataset, args.labelrate, torch.device('cuda:' + str(args.gpu)))
    if args.proxy == 1:
        proxy = torch.FloatTensor(np.vstack((sens, 1 - sens)).T)
        proxy = proxy.to(torch.device('cuda:' + str(args.gpu)))
    elif args.proxy == 2:
        proxy = torch.rand([features.shape[0], 2], requires_grad=True, device=torch.device('cuda:'+str(args.gpu)))
    else:
        proxy = None

    # create SGC model as Generator
    model = SGC(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                args.dropout)

    if labels.dim() == 1:
        loss_fcn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown dataset with wrong labels: {}'.format(args.dataset))

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr / 10, weight_decay=args.weight_decay)
    if args.proxy == 2:
        optimizer_proxy = torch.optim.Adam([proxy], lr=args.lr, weight_decay=args.weight_decay)
    Discriminator_e = local_emb_D(n_hidden=args.n_hidden)
    Discriminator_g = global_emb_D(n_hidden=args.n_hidden)
    Discriminator = logits_D(n_classes, n_hidden=n_classes)
    if cuda:
        model.cuda()
        Discriminator.cuda()
        Discriminator_e.cuda()
        Discriminator_g.cuda()
    opt_D = torch.optim.Adam([{"params": Discriminator.parameters()}, {"params": Discriminator_e.parameters()}, {"params": Discriminator_g.parameters()}],
                             lr=args.lr, weight_decay=args.weight_decay)
    loss_dis = torch.nn.BCELoss()

    param_count = sum(param.numel() for param in model.parameters()) + sum(param.numel() for param in Discriminator.parameters()) + sum(param.numel() for param in Discriminator_e.parameters()) + sum(param.numel() for param in Discriminator_g.parameters())

    dur = []
    log_every = 1
    best_eval_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred_0 = final_pred_1 = 0
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        if args.proxy in [1, 2]:
            feature = torch.concat([features, proxy.mean(axis=0).repeat([features.shape[0], 1])], 1)
        else:
            feature = features
        logits = model(feature)
        label_loss = loss_fcn(logits[train_mask], labels[train_mask])
        if args.role == 'stu':
            # ============================================
            #  Train Dis
            # ============================================
            if epoch % args.d_critic == 0:
                loss_D = 0
                ## distinguish by Dl
                Discriminator.train()
                stu_logits = logits.detach()
                pos_z = Discriminator(tea_logits)
                neg_z = Discriminator(stu_logits)
                real_z = torch.sigmoid(pos_z[:, -1])
                fake_z = torch.sigmoid(neg_z[:, -1])
                ad_loss = loss_dis(real_z, torch.ones_like(real_z)) + loss_dis(fake_z, torch.zeros_like(fake_z))
                ds_loss = loss_fcn(pos_z[:, :-1][train_mask], labels[train_mask]) + loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])
                loss_D = 0.5 * (ad_loss + ds_loss)

                # distinguish by De
                pos_e = Discriminator_e(tea_emb, g)
                neg_e = Discriminator_e(model.emb.detach(), g)
                real_e = torch.sigmoid(pos_e)
                fake_e = torch.sigmoid(neg_e)
                ad_eloss = loss_dis(real_e, torch.ones_like(real_e)) + loss_dis(fake_e, torch.zeros_like(fake_e))
                # ++++++++++++++++++++++++
                tea_sum = torch.sigmoid(tea_emb.mean(dim=0)).unsqueeze(-1)
                pos_g = Discriminator_g(tea_emb, tea_sum)
                neg_g = Discriminator_g(model.emb.detach(), tea_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss1 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))

                stu_sum = torch.sigmoid(model.emb.detach().mean(dim=0)).unsqueeze(-1)
                neg_g = Discriminator_g(tea_emb, stu_sum)
                pos_g = Discriminator_g(model.emb.detach(), stu_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss2 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))
                loss_D = loss_D + ad_eloss + ad_gloss1 + ad_gloss2
                # ++++++++++++++++++++++++

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # ============================================
            #  Train Stu
            # ============================================
            if epoch % args.g_critic == 0:
                loss_G = label_loss
                ## to fool Discriminator_l
                Discriminator.eval()
                pos_z = Discriminator(tea_logits)
                neg_z = Discriminator(logits)
                fake_z = torch.sigmoid(neg_z[:, -1])
                ad_loss = loss_dis(fake_z, torch.ones_like(fake_z))
                ds_loss = loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])  # right one
                l1_loss = torch.norm(logits - tea_logits, p=1) * 1. / len(tea_logits)
                loss_G = loss_G + 0.5 * (ds_loss + ad_loss) + l1_loss

                ## to fool Discriminator_e
                Discriminator_e.eval()
                neg_e = Discriminator_e(model.emb, g)
                fake_e = torch.sigmoid(neg_e)
                ad_eloss = loss_dis(fake_e, torch.ones_like(fake_e))
                #++++++++++++++++++++++++
                tea_sum = torch.sigmoid(tea_emb.mean(dim=0)).unsqueeze(-1)
                neg_g = Discriminator_g(model.emb, tea_sum)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss1 = loss_dis(fake_g, torch.ones_like(fake_g))

                stu_sum = torch.sigmoid(model.emb.mean(dim=0)).unsqueeze(-1)
                neg_g = Discriminator_g(tea_emb, stu_sum)
                pos_g = Discriminator_g(model.emb, stu_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss2 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))
                #++++++++++++++++++++++++
                loss_G = loss_G + ad_eloss + ad_gloss1 + ad_gloss2

                if args.proxy == 2:
                    if args.metric == 'sp':
                        loss_G = loss_G + args.alpha * approx_loss(F.softmax(logits, dim=-1), sens, idx_train) # 100
                    elif args.metric == 'eo':
                        loss_G = loss_G + args.alpha * approx_loss_eo(F.softmax(logits, dim=-1), sens, labels, idx_train) # 10
                    else:
                        raise

                optimizer.zero_grad()
                loss_G.backward()
                optimizer.step()

                if args.proxy == 2:
                    feature = torch.concat([features, proxy], 1)
                    logits = model(feature)
                    optimizer_proxy.zero_grad()
                    if args.metric == 'sp':
                        loss_G_2 = -args.alpha * approx_loss(F.softmax(logits, dim=-1), sens, torch.arange(features.shape[0])) # 100
                    elif args.metric == 'eo':
                        loss_G_2 = -args.alpha * approx_loss_eo(F.softmax(logits, dim=-1), sens, labels, torch.arange(features.shape[0])) # 10
                    else:
                        raise
                    loss_G_2.backward()
                    optimizer_proxy.step()
        else:
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_D = loss

        if epoch >= 3:
            dur.append(time.time() - t0)

        if args.proxy in [1, 2]:
            feature = torch.concat([features, proxy.mean(axis=0).repeat([features.shape[0], 1])], 1)
        else:
            feature = features
        logits = model(feature)
        val_loss = loss_fcn(logits[val_mask], labels[val_mask])
        eval_acc = compute_micro_f1(logits, labels, val_mask) 
        test_acc = compute_micro_f1(logits, labels, test_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_eval_acc = eval_acc
            final_test_acc = test_acc
            # evaluate fairness
            preds = logits.max(1)[1].type_as(labels).cpu().numpy()
            parity, equality = fair_metric(preds[idx_test.cpu().numpy()], labels[idx_test.cpu().numpy()].cpu().numpy(),
                                           sens[idx_test.cpu().numpy()])
            final_pred_0 = (preds==0).sum()
            final_pred_1 = preds.sum()
            final_parity = parity
            final_equality = equality
        # if epoch % log_every == 0:
        #     print(f"Run: {n_running}/{args.n_runs} | Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Loss {loss_D.item():.4f} | "
        #     f"Val {eval_acc:.4f} | Test {test_acc:.4f} | Best Test {final_test_acc:.4f} | ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    return best_eval_acc, final_test_acc, final_parity, final_equality
#++++++++++++++++++++++++


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    print()
    return itertools.starmap(Variant, itertools.product(*items.values()))


def main(args):
    # load and preprocess dataset
    device = torch.device('cuda:' + str(args.gpu))
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'flickr':
        from torch_geometric.datasets import Flickr
        import torch_geometric.transforms as T
        # TODO: hardcode data dir
        # root = '../../../../../datasets'
        root = '../../datasets'
        pyg_data = Flickr(root=f'{root}/Flickr', pre_transform=T.ToSparseTensor())[0]  # replace edge_index with adj
        adj = pyg_data.adj_t.to_torch_sparse_coo_tensor()
        u, v = adj.coalesce().indices()
        g = dgl.graph((u, v))

        g.ndata['feat'] = pyg_data.x
        g.ndata['label'] = pyg_data.y
        g.ndata['train_mask'] = pyg_data.train_mask
        g.ndata['val_mask'] = pyg_data.val_mask
        g.ndata['test_mask'] = pyg_data.test_mask
        n_classes = pyg_data.y.max().item() + 1
    elif args.dataset == 'reddit':
        from torch_geometric.datasets import Reddit2
        import torch_geometric.transforms as T
        # TODO: hardcode data dir
        # root = '../../../../../datasets'
        root = '../../datasets'
        pyg_data = Reddit2(f'{root}/Reddit2', pre_transform=T.ToSparseTensor())[0]
        pyg_data.x = (pyg_data.x - pyg_data.x.mean(dim=0)) / pyg_data.x.std(dim=0)
        adj = pyg_data.adj_t.to_torch_sparse_coo_tensor()
        u, v = adj.coalesce().indices()
        g = dgl.graph((u, v))
        g.ndata['feat'] = pyg_data.x
        g.ndata['label'] = pyg_data.y
        g.ndata['train_mask'] = pyg_data.train_mask
        g.ndata['val_mask'] = pyg_data.val_mask
        g.ndata['test_mask'] = pyg_data.test_mask
        n_classes = pyg_data.y.max().item() + 1
    elif args.dataset in ['bail', 'credit', 'small', 'medium']:
        adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test, sens = \
            load_tensor_data(args.teacher, args.dataset, args.labelrate, device)
        g = dgl.graph((adj_sp.row, adj_sp.col)).to(device)
        g.ndata['feat'] = features
        train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        train_mask[idx_train.to('cpu').numpy()] = True
        val_mask[idx_val.to('cpu').numpy()] = True
        test_mask[idx_test.to('cpu').numpy()] = True
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        g.ndata['label'] = labels
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        n_classes = labels.max().item() + 1
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.dataset not in ['reddit', 'yelp', 'flickr', 'corafull', 'credit', 'bail', 'DBLP', 'small', 'medium']:
        g = data[0]
        n_classes = data.num_labels
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    # normalization
    degs = g.in_degrees().clamp(min=1).float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    #++++++++++++++++++++++++
    # run
    val_accs = []
    test_accs = []
    parity = []
    equality = []

    for i in range(args.n_runs):
        val_acc, test_acc, test_parity, test_equality= run(args, g, n_classes, cuda, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        parity.append(test_parity)
        equality.append(test_equality)
        args.seed += 1

    variants = list(gen_variants(dataset=[args.dataset],
                                 model=[args.teacher],
                                 seed=[args.seed]))
    # print(variants)
    results_dict = defaultdict(list)
    for variant in variants:
        results = dict(TestAcc=np.mean(test_accs), ValAcc=np.mean(val_accs), DeltaSP=np.mean(parity), DeltaEO=np.mean(equality))
        results_dict[variant[:]] = [results]
        print("Final results:")
        output_results(results_dict, "github")


    # ----------optional : recording results----------
    f = open(f"./{args.dataset}.txt", 'a')
    s1 = f"STUDENT"
    s2 = f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}"
    s3 = f"Average test accuracy on {args.dataset}: {np.mean(test_accs)} ± {np.std(test_accs)}"
    s4 = f"Average parity on {args.dataset}: {np.mean(parity)} ± {np.std(parity)}"
    s5 = f"Average equality on {args.dataset}: {np.mean(equality)} ± {np.std(equality)}"
    f.write(s1 + '\n' + s2 + '\n' + s3 + '\n' + s4 + '\n' + s5 + '\n')

    #++++++++++++++++++++++++
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--teacher", type=str, default="GCN",
                        help="Teacher model name ('GCN', 'GraphSAGE').")
    parser.add_argument("--metric", type=str, default="sp",
                        help="Adopted fairness metric ('sp', 'eo').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=600,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--alpha", type=float, default=1e1,
                        help="Weight for fairness loss term")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument('--labelrate', type=int, default=20, help='Label rate')
    parser.add_argument("--proxy", type=int, default=2,
                        help="whether use proxy")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--role", type=str, default="stu", 
                        choices=['stu', 'vani'])
    parser.set_defaults(self_loop=True)

    parser.add_argument("--d-critic", type=int, default=1, help="train discriminator")
    parser.add_argument("--g-critic", type=int, default=1, help="train generator")
    parser.add_argument("--n-runs", type=int, default=1, help="running times")
    args = parser.parse_args()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # print(args)

    main(args)


"""
PubMed

Runned 10 times
Val Accs: [0.846, 0.844, 0.842, 0.842, 0.846, 0.842, 0.84, 0.846, 0.84, 0.846]
Test Accs: [0.807, 0.812, 0.82, 0.809, 0.818, 0.813, 0.816, 0.813, 0.813, 0.813]
Average val accuracy: 0.8433999999999999 ± 0.0023748684174075855
Average test accuracy on pubmed: 0.8134 ± 0.003666060555964638
"""