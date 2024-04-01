#!/usr/bin/env python3
from sklearn.preprocessing import normalize, StandardScaler
import math
import numpy as np
import pandas as pd
from os import path
import os
import ast
import torch.nn.functional as F
import torch.nn as nn
import torch
import bisect
from datetime import datetime


def log_message(file_name, msg, file_path='./log/', verbose=True):
    s = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' -> ' + str(msg) + '\n'
    print(s)

    with open(os.path.join(file_path, file_name), 'a+') as f:
        f.write(s)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate the modulating factor
        p_t = torch.exp(-ce_loss)
        modulating_factor = (1 - p_t) ** self.gamma

        # Calculate the focal loss
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def discretize_features_list(values, null_values=None):
    values = [str(v) for v in values]
    get_bin = lambda x, n: format(x, 'b').zfill(n)

    if null_values is None:
        null_values = []
    null_values = list(set([''] + ['nan'] + null_values))
    unique_values = set(values)
    for null_val in null_values:
        if null_val in unique_values: unique_values.remove(null_val)
    unique_values = ['null'] + sorted(list(unique_values))
    val_list = list(values)
    for null_val in null_values:
        if null_val in val_list:
            val_list = list(map(lambda x : x.replace(null_val, 'null'), val_list))
            unique_values = list(map(lambda x : x.replace(null_val, 'null'), unique_values))

    if 'null' not in val_list:
        unique_values = unique_values[1:]

    e_to_idx = {v:i for i, v in enumerate(unique_values)}
    e_to_bidx = {}

    n_bit = len("{0:b}".format(max(e_to_idx.values())))
    for idx in e_to_idx.values():
        b_idx = [int(bb) for bb in get_bin(idx, n_bit)]
        e_to_bidx[unique_values[idx]] = b_idx

    return ([e_to_bidx[val] for val in val_list]), n_bit


def normalize_features_list(values, null_values=[]):
    values_list = [val if (not math.isnan(val) or val in null_values) else 0 for val in values]
    values_list = np.array([values_list]).transpose(1,0)
    scaler = StandardScaler()
    values_list = scaler.fit_transform(X=values_list).tolist()

    return values_list


def flat_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.sqrt(np.sum(vec1**2)) * np.sqrt(np.sum(vec2**2))
    return dot_product / magnitude

def compute_similarities(train_idx, test_idx, similarity_matrix_path='./log/',
                         drugs_file='./data/datasets/base_chem/entities/drug.csv'):
    out_file = path.join(similarity_matrix_path, drugs_file.split('/')[-3] + '.npz')
    if not path.isfile(out_file):
        print(f"File {out_file} not present, computing similarities...")

        df = pd.read_csv(drugs_file)
        drugs = df['idx']
        fingerprints = [np.array(ast.literal_eval(el)) for el in df['fingerprints']]

        sim = np.zeros((len(drugs), len(drugs)))

        for i_x, drug_x in enumerate(drugs):
            for i_y, drug_y in enumerate(drugs):
                sim[drug_x, drug_y] = cosine_similarity(fingerprints[i_x], fingerprints[i_y])

        np.save(out_file, sim)
    else:
        sim = np.load(out_file)

    train = list(set(list(train_idx.col)))
    test = list(set(list(test_idx.col)))

    train_val = sorted(list(train_idx.data))

    test_val = sum(list(test_idx.data))/len(test_idx.data)

    out_perc = bisect.bisect(train_val,test_val) / len(train_val)

    sim = sim[train, :][: , test]
    maxx = sim.max(axis=0)
    avgs = sim.mean(axis=0)
    avg = sum(maxx)/len(maxx)
    print(f"Average similarity {avg}")
    return avg, out_perc, np.std(test_idx.data), maxx, avgs
