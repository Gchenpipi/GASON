import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch_sparse import SparseTensor
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
from torch import optim
from torch.nn.parameter import Parameter
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from model import *
from arguments import arg_parse
from evaluate_embedding import evaluate_embedding
from evaluate_embedding import evaluate_val,evaluate_test
from torch_geometric.transforms import Constant
import pdb
import logging
from torch.autograd import Variable
from copy import deepcopy
from CPGDataset import MultiCPGDataset
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset


LOG_FORMAT = "%(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='Accuracy.txt', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


def forward(self, x, edge_index, batch, num_graphs):
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    measure = 'JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0

    return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def gen_ran_output(data, model, vice_model, args):
    for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(
                0, torch.ones_like(param.data) * param.data.std()
            ).to(device)
    z2 = vice_model(data.x, data.edge_index, data.batch, data.num_graphs)
    return z2


class AverageMeter(object):
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)


class TwoAugUnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.transform(image), self.transform(image)

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    args = arg_parse()
    setup_seed(args.seed)
    device = torch.device(args.device)
    accuracies = {'val': [], 'test': []}
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    full_dataset = MultiCPGDataset(
        root="data/Reveal",
        label_csv="data/Reveal/Reveal_label.csv"
    )
    print(f"完整数据集大小: {len(full_dataset)}")

    import pandas as pd

    train_names = set(pd.read_csv("data/Reveal/dataset/train_set.csv")["Filename"])
    val_names = set(pd.read_csv("data/Reveal/dataset/val_set.csv")["Filename"])
    test_names = set(pd.read_csv("data/Reveal/dataset/test_set.csv")["Filename"])

    train_data_list = [d for d in full_dataset if d.sample_name in train_names]
    val_data_list = [d for d in full_dataset if d.sample_name in val_names]
    test_data_list = [d for d in full_dataset if d.sample_name in test_names]

    class SubsetDataset(InMemoryDataset):
        def __init__(self, data_list):
            super().__init__("")
            self.data, self.slices = self.collate(data_list)

    train_dataset = SubsetDataset(train_data_list)
    val_dataset = SubsetDataset(val_data_list)
    test_dataset = SubsetDataset(test_data_list)

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataset_num_features = full_dataset.num_features

    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    vice_model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        loss_all = 0
        for data in tqdm(train_loader, desc=f"Batch (Epoch {epoch})", leave=False):
            optimizer.zero_grad()
            data = data.to(device)
            x2 = gen_ran_output(data, model, vice_model, args)
            x1 = model(data.x, data.edge_index, data.batch, data.num_graphs)
            loss = model.loss_cal(x2, x1)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * data.num_graphs

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(train_loader)))

        if epoch % 20 == 0:
            model.eval()
            train_emb, train_y, _ = model.encoder.get_embeddings(train_loader)
            val_emb, val_y, _ = model.encoder.get_embeddings(val_loader)
            acc_val = evaluate_val(train_emb, train_y, val_emb, val_y)
            print(f"Val acc: {acc_val:.4f}")
            accuracies['val'].append(acc_val)

    model.eval()

    def save_embeddings(loader, split_name):
        emb, y, names = model.encoder.get_embeddings(loader)

        rows = []
        for i in range(len(emb)):
            rows.append({
                "sample_name": names[i],
                "label": int(y[i]),
                "embedding": json.dumps(emb[i].tolist())
            })
        df = pd.DataFrame(rows)
        os.makedirs("result/Reveal", exist_ok=True)
        save_emb_path = f"result/Reveal/{split_name}_embeddings.csv"
        df.to_csv(save_emb_path, index=False)
        print(f"{split_name} 嵌入已保存到 {save_emb_path}")
        return emb, y


    train_emb, train_y = save_embeddings(train_loader, "train")
    val_emb, val_y = save_embeddings(val_loader, "val")
    test_emb, test_y = save_embeddings(test_loader, "test")

    acc_test = evaluate_test(train_emb, train_y, test_emb, test_y)
    print(f"Final Test acc: {acc_test:.4f}")
    accuracies['test'].append(acc_test)

    os.makedirs("model/Reveal", exist_ok=True)
    save_model_path = "model/Reveal/saved_model.pth"
    torch.save(model.state_dict(), save_model_path)
    print(f"模型已保存到 {save_model_path}")

