#!/usr/bin/env python3

import torch as t
from NXTfusion.NXmodels import NXmodelProto
import numpy as np
import pandas as pd
import ast
import torch_geometric as pyg
from tqdm import tqdm
from scipy import sparse as sp



# Dynamic model, built on the basis of the ER Graph
class NxtDRPMC(NXmodelProto):
    def __init__(self, ERG, name, main_emb_size, gnn_size=30, out_emb_size=15, dropout=0.5, device='cuda',
                 side_info_file='./data/datasets/base_gnn/entities/drug.csv'):
        super(NxtDRPMC, self).__init__()
        self.device = device
        self.name = name

        self.entities = {}
        self.relations = set()

        print(f"Initializing model for the following relations : {ERG.lookup.keys()}")

        ACTIVATION = t.nn.Tanh
        # Net definition
        self.latent_sizes = {}


        for rel in ERG.lookup.keys():
            ent0, ent1 = rel.split('-')
            len_ent0, len_ent1 = ERG[rel]["lenDomain1"], ERG[rel]["lenDomain2"]

            self.relations.add(rel)
            self.entities[ent0] = len_ent0
            self.entities[ent1] = len_ent1
            self.latent_sizes[ent0] = main_emb_size
            self.latent_sizes[ent1] = main_emb_size

        self.__name__ = "_".join(self.entities.keys())
        self.name = "_".join(self.entities.keys())

        self.latent_sizes['drug'] = int(gnn_size * 0.8)

        self.embs = t.nn.ModuleDict()
        self.embs_h = t.nn.ModuleDict()
        self.bi_rel = t.nn.ModuleDict()
        self.out_rel = t.nn.ModuleDict()


        for entity, entity_len in self.entities.items():
            self.embs[entity] = t.nn.Embedding(entity_len, self.latent_sizes[entity])
            self.embs_h[entity] = t.nn.Sequential(t.nn.Linear(self.latent_sizes[entity],
                                                              out_emb_size),
                                       t.nn.LayerNorm(out_emb_size), ACTIVATION())
        for rel in self.relations:
            if rel == 'cell_line-drug':
                self.bi_rel[rel] = t.nn.Bilinear(out_emb_size, out_emb_size+1, out_emb_size)
            elif rel == 'drug-drug':
                self.bi_rel[rel] = t.nn.Bilinear(out_emb_size+1, out_emb_size+1, out_emb_size)
            else:
                self.bi_rel[rel] = t.nn.Bilinear(out_emb_size, out_emb_size, out_emb_size)
            if rel == 'cell_line-gene':
                # binary
                self.out_rel[rel] = t.nn.Sequential(t.nn.LayerNorm(out_emb_size), ACTIVATION(),
                                            t.nn.Dropout(0.1),
                                                    t.nn.Linear(out_emb_size, 1),
                                                    t.nn.Sigmoid())
            else:
                self.out_rel[rel] = t.nn.Sequential(t.nn.LayerNorm(out_emb_size), ACTIVATION(),
                                            t.nn.Dropout(0.1),
                                            t.nn.Linear(out_emb_size, 1))
        self.dropout = dropout

        self.gnn = pyg.nn.Sequential('x, edge_index, batch', [
            (pyg.nn.GATv2Conv(in_channels=51, out_channels=gnn_size), 'x, edge_index -> x'),
            (pyg.nn.BatchNorm(in_channels=gnn_size), 'x -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.GATv2Conv(in_channels=gnn_size, out_channels=gnn_size,
                              dropout=self.dropout), 'x, edge_index -> x'),
            (pyg.nn.BatchNorm(in_channels=gnn_size), 'x -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.GATv2Conv(in_channels=gnn_size, out_channels=gnn_size,
                              dropout=self.dropout), 'x, edge_index -> x'),
            (pyg.nn.BatchNorm(in_channels=gnn_size), 'x -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.GATv2Conv(in_channels=gnn_size, out_channels=gnn_size,
                              dropout=self.dropout), 'x, edge_index -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.GATv2Conv(in_channels=gnn_size,
                              out_channels=int(gnn_size*0.8)), 'x, edge_index -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.global_add_pool, 'x, batch-> x')
        ])

        print("Gather graph properties")
        self.drugs_features = self.get_side_info_chemical(np.arange(
            ERG['cell_line-drug']['lenDomain2']), side_info_file=side_info_file)
        x, edges = zip(*self.drugs_features)

        graphs = []
        # self.conc = []
        for i in range(len(x)):
            x_i =  x[i] #[x_ii for x_ii in x[i]]
            data = pyg.data.Data(t.tensor(x_i, dtype=t.float, device=self.device),
                                t.tensor(edges[i], dtype=t.long, device=self.device).transpose(1,0),
                                device=self.device)
            graphs.append(data)
            # self.conc.append(maxconc[i])
        self.graphs = graphs

        self.max_conc = self.get_max_conc()
        self.batches = {}
        self.apply(self.init_weights)
        self.to(device)
        

    def get_max_conc(self):
        mc = sp.load_npz('./data/datasets/base_auc/relations/drug_response_max_conc.npz')
        mc = mc.todense()

        fun_transf = lambda x : 1 / (1 + ((x+0.000001)**(-0.1)))
        # apply to all elements of matrix
        mc = np.vectorize(fun_transf)(mc)

        return mc


    def get_side_info_chemical(self, indices, side_info_file='./data/datasets/base_gnn/entities/drug.csv'):
        '''
        Get side information numpy array
        '''
        print("Get chemical")
        df = pd.read_csv(side_info_file)
        features = df.set_index('idx')
        features = features[['atomic_features', 'atomic_bonds']]

        out_f = []
        for i in indices:
            row = features.loc[i]
            rl = row.to_list()
            rll = [ast.literal_eval(e) for e in rl]
            rll = rll #+ [rl[-1]]
            out_f.append(rll)
        return out_f


    def forward(self, rel, i1, i2, s1=None, s2=None):
        entities = rel.split('-')
        indices = [i1, i2]

        rel_idx = np.array([indices[0].tolist(), indices[1].tolist()])
        embs = []

        # Forward embs
        for i, ent in enumerate(entities):
            # Drug emb
            if ent == 'drug':
                ii , ii_mapping = t.unique(indices[i], sorted=True, return_inverse=True)

                if str(ii.tolist()) not in self.batches:
                    drugs_info = [self.graphs[idx] for idx in ii]
                    data_batched = pyg.data.Batch.from_data_list(drugs_info)
                    self.batches[str(ii.tolist())] = data_batched
                else:
                    data_batched = self.batches[str(ii.tolist())]

                x  = self.gnn(data_batched.x, data_batched.edge_index,
                                batch=data_batched.batch)
               
                x = self.embs_h[ent](x)
                x = x[ii_mapping]
                conc = t.tensor(self.max_conc[rel_idx[0], rel_idx[1]], dtype=t.float,
                                device=self.device).transpose(1,0)
                x = t.concat((x, conc), dim=1)
                embs.append(x)
            else:
                x = self.embs[ent](indices[i])
                x = self.embs_h[ent](x)
                embs.append(x)
        assert len(embs) == 2
        # Gen relation out
        o = self.bi_rel[rel](*embs)
        o = self.out_rel[rel](o)

        return o


    def get_latent_repr(self, rel):
        e0, e1 = rel.split('-')

        e0_indices = list(range(self.entities[e0]))
        e1_indices = list(range(self.entities[e1]))

        e1_latent = []
        if e1 == 'drug':
            for bs in range(0, len(e1_indices), 1000):
                ii = e1_indices[bs:bs+1000]
                drugs_info = [self.graphs[idx] for idx in ii]
                data_batched = pyg.data.Batch.from_data_list(drugs_info)
                x  = self.gnn(data_batched.x, data_batched.edge_index,
                                batch=data_batched.batch)
                e1_latent.append(x)
            e1_latent = t.cat(e1_latent, dim=0)
        else:
            e1_latent = self.embs[e1](t.tensor(e1_indices, dtype=t.long, device=self.device))
            e1_latent = self.embs_h[e1](e1_latent)

        e0_latent = self.embs[e0](t.tensor(e0_indices, dtype=t.long, device=self.device))
        e0_latent = self.embs_h[e0](e0_latent)
        return e0_latent, e1_latent


class NxtDRP(NXmodelProto):
    def __init__(self, ERG, name, main_emb_size, gnn_size=30, out_emb_size=15, dropout=0.5, device='cuda',
                 side_info_file='./data/datasets/base_gnn/entities/drug.csv'):
        super(NxtDRP, self).__init__()
        self.device = device
        self.name = name

        self.entities = {}
        self.relations = set()

        print(f"Initializing model for the following relations : {ERG.lookup.keys()}")

        ACTIVATION = t.nn.Tanh
        # Net definition
        self.latent_sizes = {}


        for rel in ERG.lookup.keys():
            ent0, ent1 = rel.split('-')
            len_ent0, len_ent1 = ERG[rel]["lenDomain1"], ERG[rel]["lenDomain2"]

            self.relations.add(rel)
            self.entities[ent0] = len_ent0
            self.entities[ent1] = len_ent1
            self.latent_sizes[ent0] = main_emb_size
            self.latent_sizes[ent1] = main_emb_size

        self.__name__ = "_".join(self.entities.keys())
        self.name = "_".join(self.entities.keys())

        self.latent_sizes['drug'] = int(gnn_size * 0.8)

        self.embs = t.nn.ModuleDict()
        self.embs_h = t.nn.ModuleDict()
        self.bi_rel = t.nn.ModuleDict()
        self.out_rel = t.nn.ModuleDict()


        for entity, entity_len in self.entities.items():
            self.embs[entity] = t.nn.Embedding(entity_len, self.latent_sizes[entity])
            self.embs_h[entity] = t.nn.Sequential(t.nn.Linear(self.latent_sizes[entity], out_emb_size),
                                       t.nn.LayerNorm(out_emb_size), ACTIVATION())
        for rel in self.relations:
            self.bi_rel[rel] = t.nn.Bilinear(out_emb_size, out_emb_size, out_emb_size)
            if rel == 'cell_line-gene':
                # binary
                self.out_rel[rel] = t.nn.Sequential(t.nn.LayerNorm(out_emb_size), ACTIVATION(),
                                            t.nn.Dropout(0.1),
                                                    t.nn.Linear(out_emb_size, 1),
                                                    t.nn.Sigmoid())
            else:
                self.out_rel[rel] = t.nn.Sequential(t.nn.LayerNorm(out_emb_size), ACTIVATION(),
                                            t.nn.Dropout(0.1),
                                            t.nn.Linear(out_emb_size, 1))

        self.dropout = dropout

        self.gnn = pyg.nn.Sequential('x, edge_index, batch', [
            (pyg.nn.GATv2Conv(in_channels=51, out_channels=gnn_size), 'x, edge_index -> x'),
            (pyg.nn.BatchNorm(in_channels=gnn_size), 'x -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.GATv2Conv(in_channels=gnn_size, out_channels=gnn_size,
                              dropout=self.dropout), 'x, edge_index -> x'),
            (pyg.nn.BatchNorm(in_channels=gnn_size), 'x -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.GATv2Conv(in_channels=gnn_size, out_channels=gnn_size,
                              dropout=self.dropout), 'x, edge_index -> x'),
            (pyg.nn.BatchNorm(in_channels=gnn_size), 'x -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.GATv2Conv(in_channels=gnn_size, out_channels=gnn_size,
                              dropout=self.dropout), 'x, edge_index -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.GATv2Conv(in_channels=gnn_size,
                              out_channels=int(gnn_size*0.8)), 'x, edge_index -> x'),
            t.nn.ReLU(inplace=True),
            (pyg.nn.global_add_pool, 'x, batch-> x')
        ])

        print("Gather graph properties")
        self.drugs_features = self.get_side_info_chemical(np.arange(
            ERG['cell_line-drug']['lenDomain2']), side_info_file=side_info_file)
        x, edges = zip(*self.drugs_features)

        graphs = []
        for i in range(len(x)):
            x_i =  x[i] #[x_ii for x_ii in x[i]]
            data = pyg.data.Data(t.tensor(x_i, dtype=t.float, device=self.device),
                                t.tensor(edges[i], dtype=t.long, device=self.device).transpose(1,0),
                                device=self.device)
            graphs.append(data)
        self.graphs = graphs



        self.batches = {}
        self.apply(self.init_weights)
        self.to(device)


    def get_side_info_chemical(self, indices, side_info_file='./data/datasets/base_gnn/entities/drug.csv'):
        '''
        Get side information numpy array
        '''
        print("Get chemical")
        df = pd.read_csv(side_info_file)
        features = df.set_index('idx')
        features = features[['atomic_features', 'atomic_bonds']]

        out_f = []
        for i in indices:
            row = features.loc[i]
            rl = row.to_list()
            rll = [ast.literal_eval(e) for e in rl]
            rll = rll
            # rl = flat_list(rl)
            out_f.append(rll)
        # out_f = np.array(out_f)
        return out_f


    def forward(self, rel, i1, i2, s1=None, s2=None):
        entities = rel.split('-')
        indices = [i1, i2]

        embs = []

        # Forward embs
        for i, ent in enumerate(entities):
            # Drug emb
            if ent == 'drug':
                ii, ii_mapping = t.unique(indices[i], sorted=True, return_inverse=True)

                if str(ii.tolist()) not in self.batches:
                    drugs_info = [self.graphs[idx] for idx in ii]
                    data_batched = pyg.data.Batch.from_data_list(drugs_info)
                    self.batches[str(ii.tolist())] = data_batched
                else:
                    data_batched = self.batches[str(ii.tolist())]

                x  = self.gnn(data_batched.x, data_batched.edge_index,
                                batch=data_batched.batch)
                #conc = t.tensor([[self.conc[idx]] for idx in ii], dtype=t.float, device=self.device)
                #x = t.concat((x, conc), dim=1)
                x = self.embs_h[ent](x)
                x = x[ii_mapping]
                embs.append(x)
            else:
                x = self.embs[ent](indices[i])
                x = self.embs_h[ent](x)
                embs.append(x)
        assert len(embs) == 2

        # Gen relation out
        o = self.bi_rel[rel](*embs)
        o = self.out_rel[rel](o)

        return o
