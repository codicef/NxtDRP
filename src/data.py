#!/usr/bin/env python3
from pprint import pprint
import pandas as pd
import numpy as np
from os import path, mkdir, listdir, makedirs
import pprint
from NXTfusion import NXTfusion as NX, NXFeaturesConstruction as NFeat
from NXTfusion import DataMatrix as DM
from sklearn.metrics import accuracy_score
from utils import discretize_features_list, normalize_features_list, flat_list
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import sparse
from operator import itemgetter
import ast
import math
import seaborn as sns
from matplotlib import pyplot as plt
import collections
import json



class DatasetHandler:
    '''
    Dataset handler class

    Args:
        relations: List of relation objects
        serialize_path: Path to serialize the dataset
        overwrite: Overwrite the existing serialized dataset

    '''
    def __init__(self,
                 relations=[],
                 serialize_path="./data/datasets/base/",
                 overwrite=False,
                 seed=1956,
                 ):

        self.serialize_path = serialize_path
        self.seed = seed  # random state of the train/test splits
        self.rel_dict = {}
        self.relations = []
        self.entities = []
        self.entities_dict = {}
        # Update all indices
        for rel in relations:
            self.rel_dict[rel.name] = rel
            if serialize_path is not None:
                rel.entity_0.update_indices(rel.e0_keys)
                rel.entity_1.update_indices(rel.e1_keys)


            self.relations.append(rel)
            if rel.entity_0 not in self.entities:
                self.entities.append(rel.entity_0)
                self.entities_dict[rel.entity_0.name] = rel.entity_0
            if rel.entity_1 not in self.entities:
                self.entities.append(rel.entity_1)
                self.entities_dict[rel.entity_1.name] = rel.entity_1

        if serialize_path is not None:
            # Serialize
            if not path.isdir(serialize_path):
                makedirs(serialize_path)
            if not path.isdir(path.join(serialize_path, 'relations')):
                makedirs(path.join(serialize_path, 'relations'))
            if not path.isdir(path.join(serialize_path, 'entities')):
                makedirs(path.join(serialize_path, 'entities'))


            for rel in relations:
                # Rebuild the matrix with the final entity indices before saving
                rel._set_rel_matrix()
                rel.save(path.join(serialize_path, 'relations'))

                if overwrite or not path.isfile(path.join(serialize_path, 'entities',
                                                   rel.entity_0.name + '.csv')):
                    rel.entity_0.save(path.join(serialize_path, 'entities'))
                    assert len(rel.entity_0.e_idx.keys()) == rel.matrix.shape[0]

                # save entities
                if overwrite or not path.isfile(path.join(serialize_path, 'entities',
                                                   rel.entity_1.name + '.csv')):
                    rel.entity_1.save(path.join(serialize_path, 'entities'))

                    assert len(rel.entity_1.e_idx.keys()) == rel.matrix.shape[1]



    def get_er_graph(self, losses_dict, target_rel=None, target_data=None):
        out_rel = []  # Nx MetaRelations

        side_info_d = {}
        to_filter = int(target_rel is not None) + int(target_data is not None)
        assert to_filter != 1
        # Out relations
        for rel in self.relations:
            # infer prediction type
            if 'float' in str(rel.dtype):
                pred_type = 'regression'
            else:
                pred_type = 'binary'

            e0 = NX.Entity(rel.entity_0.name, sorted(rel.entity_0.e_idx.values()),
                           dtype=np.int32)
            e1 = NX.Entity(rel.entity_1.name, sorted(rel.entity_1.e_idx.values()),
                           dtype=np.int32)
            # e1 = NX.Entity(rel.entity_1.name, list(range(
            #     len(rel.entity_1.idx_e))), dtype=np.int32)

            if rel.entity_0.name not in side_info_d:
                if rel.entity_0.side_info_features is not None:
                    if 'drug' in rel.entity_0.name:
                        sf = DM.SideInfo(rel.entity_0.name + '_side', e0,
                                     rel.entity_0.get_side_info_chemical(sorted(
                                         rel.entity_0.e_idx.values())))
                    else:
                        sf = DM.SideInfo(rel.entity_0.name + '_side', e0,
                                     rel.entity_0.get_side_info(sorted(
                                         rel.entity_0.e_idx.values())))
                else:
                    sf = None
                side_info_d[rel.entity_0.name] = sf
            if rel.entity_1.name not in side_info_d:
                if rel.entity_1.side_info_features is not None:
                    if 'drug' in rel.entity_1.name:
                        sf = DM.SideInfo(rel.entity_1.name + '_side', e1,
                                     rel.entity_1.get_side_info_chemical(sorted(
                                         rel.entity_1.e_idx.values())))
                    else:
                        sf = DM.SideInfo(rel.entity_1.name + '_side', e0,
                                     rel.entity_1.get_side_info(sorted(
                                         rel.entity_1.e_idx.values())))
                else:
                    sf = None
                side_info_d[rel.entity_1.name] = sf


            if to_filter > 1 and target_rel.name == rel.name:  # filter for cv
                matrix = rel.matrix.toarray()
                row, col = zip(*target_data)
                data = matrix[row, col]
                matrix = sparse.coo_matrix((data, (row, col)), shape=matrix.shape)
                out_matrix = matrix

            else:
                matrix = rel.matrix

            nxmatrix = DM.DataMatrix(rel.name + '_matrix', e0, e1, matrix)
            # rel_w = 2 if rel.name == 'cell_line-drug' else 1
            nxrel = NX.Relation(rel.name, e0, e1,
                                nxmatrix, pred_type,
                                losses_dict[rel.name],
                                relationWeight=1)
            meta_rel = NX.MetaRelation(rel.name, e0, e1, relations=[nxrel],
                                       side1=side_info_d[e0.name],
                                       side2=side_info_d[e1.name])
            if to_filter > 1:
                out_target_rel = nxrel
            out_rel.append(meta_rel)

        er = NX.ERgraph(out_rel)
        if to_filter > 1:
            return er, out_matrix, out_target_rel, side_info_d
        else:
            return er, out_rel, None, side_info_d



    def set_stratified_folds(self, target_relation, split_type='valid', stratify_group='row',
                                n_splits=None, test_indices=None):
        rel = self.rel_dict[target_relation.name]

        if stratify_group == 'row':
            row_values = list(rel.matrix.row)
            row_keys = list(set(rel.matrix.row))
            row_keys = {k:i for i, k in enumerate(row_keys)}
            groups = [row_keys[v] for v in row_values]
        elif stratify_group == 'col':
            col_values = list(rel.matrix.col)
            col_keys = list(set(rel.matrix.col))
            col_keys = {k:i for i, k in enumerate(col_keys)}
            groups = [col_keys[v] for v in col_values]


        if split_type == 'valid':
            self.valid_folds = []
            self.valid_active_indices = list(zip(list(rel.matrix.row), list(rel.matrix.col)))
            if test_indices is not None:
                to_remove = [self.valid_active_indices.index(test_val) for test_val in test_indices]
                for i_r in sorted(to_remove, reverse=True):
                    del self.valid_active_indices[i_r]
                    del groups[i_r]
                # self.valid_active_indices = list(set(self.valid_active_indices) - set(test_indices))
            active_indices = self.valid_active_indices
        elif split_type == 'test':
            self.test_folds = []
            self.test_active_indices = list(zip(list(rel.matrix.row), list(rel.matrix.col)))
            active_indices = self.test_active_indices
        else:
            assert False


        if n_splits is None:
            n_splits = len(set(groups))
        # self.n_splits = n_splits
        kf = GroupShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=self.seed)
        # kf = LeaveOneGroupOut()

        for train_idx, valid_idx in kf.split(X=active_indices, groups=groups):
            if split_type == 'valid':
                self.valid_folds.append((train_idx, valid_idx))
            else:
                self.test_folds.append((train_idx, valid_idx))
        print(f"Stratified ({stratify_group}) cross validation indices saved into the dataset obj")
        return n_splits


    def set_cell_folds(self, target_relation, n_splits=5, split_type='valid', test_indices=None,
                       stratify_group='cell', fixed_test_indices=None):
        rel = self.rel_dict[target_relation.name]
        if split_type == 'valid':
            self.valid_folds = []
            self.valid_active_indices = list(zip(list(rel.matrix.row), list(rel.matrix.col)))
            if test_indices is not None:
                to_remove = [self.valid_active_indices.index(test_val) for test_val in test_indices]
                for i_r in sorted(to_remove, reverse=True):
                    del self.valid_active_indices[i_r]
                # self.valid_active_indices = list(set(self.valid_active_indices) - set(test_indices))
            active_indices = self.valid_active_indices
        elif split_type == 'test':
            self.test_folds = []
            self.test_active_indices = list(zip(list(rel.matrix.row), list(rel.matrix.col)))
            active_indices = self.test_active_indices


        test_size = 0.1 if stratify_group == 'cell' else 10
        kf = ShuffleSplit(n_splits=n_splits,  random_state=self.seed, test_size=test_size)


        for train_idx, test_idx in kf.split(active_indices):
            if stratify_group == 'double':
                active_indices = np.array(active_indices)
                train = active_indices[train_idx]
                test = active_indices[test_idx]
                test_rows, test_cols = list(zip(*test))
                train_rows, train_cols = list(zip(*train))

                to_delete_row = np.isin(train_rows, test_rows)
                to_delete_col = np.isin(train_cols, test_cols)

                to_delete = np.logical_not(np.logical_or(to_delete_row, to_delete_col))
                train_idx = train_idx[to_delete]
            if split_type == 'valid':
                self.valid_folds.append((train_idx, test_idx))
            else:
                if fixed_test_indices is not None:
                    train_idx, test_idx = list(set(list(active_indices))-set(fixed_test_indices)), fixed_test_indices
                    test_active_indices_idx = {indice: i for i, indice in enumerate(self.test_active_indices)}
                    train_idx = [test_active_indices_idx[test_val] for test_val in train_idx]
                    test_idx = [test_active_indices_idx[test_val] for test_val in test_idx]
                    # train_idx = [self.test_active_indices.index(test_val) for test_val in train_idx]
                    # test_idx = [self.test_active_indices.index(test_val) for test_val in test_idx]
                    print(f"Number of cell line in train :{len(set(rel.matrix.row[train_idx]))}")
                    print(f"Number of cell line in test :{len(set(rel.matrix.row[test_idx]))}")
                    print(f"Total number of cell line :{len(set(rel.matrix.row))}")
                self.test_folds.append((train_idx, test_idx))
        print("Cross validation indices saved into the dataset obj")
        return n_splits


    def get_cell_cv_folds(self, target_relation, losses_dict,
                          cv_fold, n_splits=5, cv_type='cell',
                          split_type='valid', fixed_test_indices=None):
        print("CV Split Type :" + split_type)

        rel = self.rel_dict[target_relation.name]
        if not hasattr(self, str(split_type) + '_folds'):
            if cv_type == 'cell' or cv_type == 'double':
                self.set_cell_folds(target_relation, n_splits,
                                    split_type=split_type,
                                    stratify_group=cv_type,
                                    fixed_test_indices=fixed_test_indices)

            elif cv_type in ['row', 'col']:
                self.set_stratified_folds(target_relation,
                                          stratify_group=cv_type, n_splits=n_splits,
                                          split_type=split_type)
            else:
                print("Invalid cv split type")
                assert False

        if split_type == 'valid':
            train_idx, test_idx = self.valid_folds[cv_fold]
            train_tidx = itemgetter(*train_idx)(self.valid_active_indices)
            test_tidx = itemgetter(*test_idx)(self.valid_active_indices)
        elif split_type == 'test':
            train_idx, test_idx = self.test_folds[cv_fold]

            train_tidx = itemgetter(*train_idx)(self.test_active_indices)
            test_tidx = itemgetter(*test_idx)(self.test_active_indices)
        else:
            assert False

        # if split_type == 'test':
        #     if cv_type == 'cell':
        #         self.set_cell_folds(target_relation, n_splits,
        #                             split_type='valid', test_indices=test_tidx)

        #     elif cv_type in ['row', 'col']:
        #         self.set_stratified_folds(target_relation,
        #                                   stratify_group=cv_type, n_splits=n_splits,
        #                                   split_type='valid', test_indices=test_tidx)


        self._check_overlapping_stratification(train_tidx, test_tidx, cv_type=cv_type)


        er_train, coo_matrix, nx_target_rel, side_info_d = \
            self.get_er_graph(losses_dict,
                              target_rel=target_relation,
                              target_data=train_tidx)
        test_matrix = rel.matrix.toarray()
        row, col = zip(*test_tidx)
        data = test_matrix[row, col]
        test_matrix = sparse.coo_matrix((data, (row, col)),
                                        shape=test_matrix.shape)

        return er_train, coo_matrix, test_matrix, nx_target_rel, side_info_d


    def _check_overlapping_stratification(self, train_idx, test_idx, cv_type='row'):
        tr_row, tr_col = zip(*train_idx)
        te_row, te_col = zip(*test_idx)

        if cv_type == 'row':
            assert len(set(tr_row).intersection(set(te_row))) == 0
        elif cv_type == 'col':
            assert len(set(tr_col).intersection(set(te_col))) == 0
        elif cv_type == 'cell':
            pass
        else:
            print("Invalid cv type")

    
    @staticmethod
    def load_serialized(serialized_path, load_side_info=True, relations_to_load=None):
        assert path.isdir(path.join(serialized_path, 'relations')) and path.isdir(
            path.join(serialized_path, 'entities'))

        ent_d = {}
        for ent_file in sorted(listdir(path.join(serialized_path, 'entities'))):
            entity = Entity.load_saved(path.join(serialized_path, 'entities', ent_file),
                                       load_side_info=load_side_info)
            ent_d[entity.name] = entity


        relations = []
        for rel_file in sorted(listdir(path.join(serialized_path, 'relations'))):
            if '-' not in rel_file:
                continue
            if 'idx' in rel_file:
                continue
            # rel_file_name = str(rel_file)
            # rel_file_name.replace('.npy', '')
            ee = rel_file.replace('.npz', '').split('-')
            e0, e1 = ee[0], ee[1]
            if relations_to_load is not None and f"{e0}-{e1}" not in relations_to_load:
                continue
            rel = Relation.load_saved(path.join(serialized_path,
                                                'relations', rel_file),
                                      ent_d[e0], ent_d[e1])
            relations.append(rel)

        print(f"Dataset stored at {serialized_path} loaded correctly")
        ds = DatasetHandler(relations, serialize_path=None)
        ds.serialize_path=serialized_path
        return ds


class Relation:
    def __init__(self,
                 entity_0,
                 entity_1,
                 value_key=None,
                 relation_data_path=None,
                 relation_matrix=None,
                 dtype=np.float32,
                 filter_query=None,
                 ignore_index=-1,
                 lite_load=False,
                 transform_fun=None,
                 remap_indices=None):
        """
        Relation class constructor

        Args:
            relation_path: Path of relation csv file. (rows : e0,e1,v0,v1,...)
            entity_0: First entity object (with or without side info)
            entity_1: Second entity object (with or without side info)
            value_key: csv column name to be used as relation obj value
            dtype: value data type(optional)
            filter_query: pandas custom dataset query to filter data (optional)

        """
        self.entity_0 = entity_0
        self.entity_1 = entity_1
        self.dtype = dtype
        self.ignore_index = ignore_index

        if relation_data_path is not None:
            df = pd.read_csv(relation_data_path)
            df['idx'] = np.arange(len(df))
            print(f"Relation file loaded, columns : {df.columns}")
            if entity_0.key == entity_1.key:
                e0_key = entity_0.key + '_1'
                e1_key = entity_1.key + '_2'
            else:
                e0_key, e1_key = entity_0.key, entity_1.key
            # Discard rows with missing entities or values (other columns are ignored)
            df = df.dropna(subset=[c for c in [e0_key, e1_key, value_key] if c in df.columns])
            if filter_query is not None:
                print(f"Filtering relation data with query : {filter_query}")
                print(f"Original shape : {df.shape}")
                df.query(filter_query, inplace=True)
                print(f"Filtered, new shape : {df.shape}")
            if hasattr(entity_0, 'to_skip'):
                df = df[~df[e0_key].isin(entity_0.to_skip)]
            if hasattr(entity_1, 'to_skip'):
                df = df[~df[e1_key].isin(entity_1.to_skip)]
            df[e0_key] = df[e0_key].str.lower()
            df[e1_key] = df[e1_key].str.lower()
            df = df.drop_duplicates([e0_key, e1_key], keep='last')

            self.e0_keys = df[e0_key].astype('str').str.lower().tolist()
            self.e1_keys = df[e1_key].astype('str').str.lower().tolist()
            # self.e1_keys = df[e1_key].str.lower().tolist()
            if value_key != 'binary':
                self.values = df[value_key].to_numpy(dtype=dtype)
            else:
                # TODO
                self.values = np.ones(len(df))

            if transform_fun is not None:
                new_values = transform_fun(self.values)
                assert len(new_values) == len(self.values)
                self.values = new_values


            entity_0.update_indices(self.e0_keys)
            entity_1.update_indices(self.e1_keys)

            self.matrix = self.get_rel_matrix(update_matrix=True)

            if remap_indices is not None:
                df['nidx'] = np.arange(len(df))

                self.remapped = df.loc[df['idx'].isin(remap_indices), 'nidx']

        elif relation_matrix is not None:
            self.matrix = relation_matrix.copy()

            if not lite_load:
                self.e0_keys = self.entity_0.idx_e[self.matrix.row]
                self.e1_keys = self.entity_1.idx_e[self.matrix.col]
                self.values = self.matrix.data
        else:
            assert False


        print(f"Relation matrix shape : {self.matrix.shape}")

        self.name = self.entity_0.name + '-' + self.entity_1.name #+ '_' + value_key


    def _set_rel_matrix(self, ignore_index=-1):
        e0_indices = [self.entity_0.e_idx[e] for e in self.e0_keys]
        e1_indices = [self.entity_1.e_idx[e] for e in self.e1_keys]

        vals = list(zip(e0_indices, e1_indices))
        assert len(vals) ==  len(set(vals))
        matrix = sparse.coo_matrix((self.values, (e0_indices, e1_indices)),
                                   shape=(len(self.entity_0.idx_e), len(self.entity_1.idx_e)),
                                   dtype=self.dtype)

        self.matrix = matrix

    def get_rel_matrix(self, ignore_index=None, update_matrix=False):
        if ignore_index is None:
            ignore_idx = self.ignore_index
        else:
            ignore_idx = ignore_index
        if update_matrix:
            self._set_rel_matrix(ignore_idx)
        return self.matrix.copy()

    def get_stats(self):
        stats = {}
        stats['name'] = self.name
        stats['matrix_shape'] = self.matrix.shape
        stats['sparsity'] = 1 - self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1])
        stats['n_points'] = self.matrix.nnz
        return stats

    def print_stats(self):
        pprint.pprint(self.get_stats())

    def save(self, path_dir):
        rel_matrix = self.get_rel_matrix(self)

        print(f"Serializing relations matrix (shape={rel_matrix.shape}) at {path_dir}")
        sparse.save_npz(path.join(path_dir, self.name), rel_matrix)
        print(f"Successfully serialized")

        if hasattr(self, 'remapped'):
            self.remapped.to_csv(path.join(path_dir, self.name + '_idx.csv'), index=False)
            print("Saved remapped test indices")


    @staticmethod
    def load_saved(path, entity_0, entity_1, name=None, lite_load=True):
        if name is None:
            name = path.split('/')[-1].replace('.npz', '')
        matrix = sparse.load_npz(path)

        rel = Relation(entity_0, entity_1, relation_matrix=matrix, dtype=matrix.dtype,
                       lite_load=True)

        print(f"Relation matrix {name} loaded correctly (path={path})")
        print(f"Summary of relation {name} : {rel.print_stats()}")
        return rel

class Entity:
    def __init__(self, name:str, side_info:str=None, side_info_features:list=None, entity_key:str=None,
                 side_info_transf_funs:list=None):
        self.name = name

        # Active indices list
        self.e_idx = {}
        self.side_info_features = side_info_features
        self.key = entity_key
        # Add side info
        if side_info is not None and side_info_features is not None:
            side_df = pd.read_csv(side_info)
            if self.key is None:
                self.key = side_df.columns.tolist()[0]
            side_df[self.key] = side_df[self.key].astype('str').str.lower()
            self.df = pd.DataFrame(columns=[self.key, 'idx'])
            side_df = side_df.filter(items=[self.key] + side_info_features)
            if side_info_transf_funs is not None:
                for side_key in side_info_transf_funs.keys():
                    transf = side_info_transf_funs[side_key](side_df[side_key].tolist())
                    side_df[side_key] = transf

                to_skip = side_df[side_df.isnull().any(axis=1)][self.key]
                self.to_skip = to_skip
                # self.update_indices(self.df[~self.df['idx'].isin(to_skip)][self.key],
                #                       overwrite=True)
            self.df = self.df.merge(side_df, on=self.key, how='right')

        else:
            self.df = pd.DataFrame(columns=[self.key, 'idx'])
        print(f"Entity {name} created, columns : {self.df.columns}")
        print(f"Entity {name} created, shape : {self.df.shape}")


    def get_side_info(self, indices):
        '''
        Get side information numpy array
        '''

        # if hasattr(self, 'to_skip'):
        #     aa = self.df[(~self.df['idx'].isnull()) & (~self.df['idx'].isin(self.to_skip))]
        # else:
        #     aa = self.df[~self.df['idx'].isnull()]
        features = self.df[self.side_info_features + ['idx']]
        features = features.set_index('idx')

        out_f = []
        for i in indices:
            row = features.loc[i]
            rl = row.to_list()
            rl = [ast.literal_eval(e) for e in rl]
            rl = flat_list(rl)
            out_f.append(rl)

        out_f = np.array(out_f)
        return out_f

    def get_side_info_chemical(self, indices):
        '''
        Get side information numpy array
        '''

        features = self.df[self.side_info_features + ['idx']]
        features = features.set_index('idx')

        out_f = []
        for i in indices:
            row = features.loc[i]
            rl = row.to_list()
            rl = [ast.literal_eval(e) for e in rl]
            # rl = flat_list(rl)
            print(rl)
            out_f.append(rl)

        # out_f = np.array(out_f)
        return out_f


    def get_indices(self,):
        return np.arange(len(self.idx_e), dtype=np.int16)

    def update_indices(self, values, overwrite=False):
        if not overwrite:
            m_i = set(self.e_idx.keys())
        else:
            m_i = set([])
        values = [str(a).lower() for a in values]
        uni_e = set(values)
        all_active = sorted(m_i.union(uni_e))
        self.idx_e = np.array(list(all_active))
        self.e_idx = {e:i for i, e in enumerate(all_active)}
        indices_df = pd.DataFrame({'idx': list(range(len(self.idx_e))),
                                   self.key: self.idx_e})

        self.df = self.df.drop(labels='idx', axis=1)
        self.df = indices_df.merge(self.df, on=self.key, how='left')
        self.max_idx = len(all_active)



    def save(self, path_dir):
        self.df = self.df[~self.df['idx'].isnull()]
        self.df = self.df.drop_duplicates(subset=['idx']) # TODO
        self.df.to_csv(path.join(path_dir, self.name + '.csv'), index=False)
        print("Saved")


    @staticmethod
    def load_saved(path, name=None, load_side_info=True, skip_empty=True):
        if name is None:
            name = path.split('/')[-1].replace('.csv', '')

        df = pd.read_csv(path)
        entity = Entity(name=name)
        entity.df = df
        entity.key = df.columns.tolist()[1]
        df[entity.key] = df[entity.key].astype('str').str.lower()
        entity.update_indices(df[entity.key].astype('str').str.lower().tolist())

        if len(df.columns) > 2 and (load_side_info or skip_empty):
            #entity.df[~entity.df[entity.key].isin(to_skip)]
            if load_side_info:
                entity.side_info_features = df.columns.tolist()[2:]

            to_skip = entity.df[entity.df.isnull().any(axis=1)][entity.key]
            entity.to_skip = to_skip
        else:
            entity.side_info_features = None

        print(f"Entity {name} loaded correctly (path={path})")
        return entity




# Raw input files of each dataset, relative to the raw data folder
DATASETS = {
    'gdsc': {
        'target': 'IC50',
        'transform': 'sigmoid',
        'response': 'relations/gdsc_drug_cellline_v6.csv',
        'response_filter': 'rmse <= 0.3',
        'drugs': 'entities/drugs_v6.csv',
        'rnaseq': 'relations/rnaseq_tpm_cellline_v6_top1000.csv',
        'proteomics': 'relations/protein_zscore_cellline_v6_l.csv',
    },
    # Same data, Area Under the Dose-Response Curve as target
    'gdsc_auc': {
        'target': 'auc',
        'transform': 'none',
        'response': 'relations/gdsc_drug_cellline_v6.csv',
        'response_filter': 'rmse <= 0.3',
        'drugs': 'entities/drugs_v6.csv',
        'rnaseq': 'relations/rnaseq_tpm_cellline_v6_top1000.csv',
        'proteomics': 'relations/protein_zscore_cellline_v6_l.csv',
    },
    'ccle': {
        'target': 'IC50',
        'transform': 'sigmoid',
        'response': 'relations/ccle_drug_response.csv',
        'response_filter': None,
        'drugs': 'entities/drugs_ccle.csv',
        'rnaseq': 'relations/ccle_rnaseq_top500.csv',
        'proteomics': 'relations/ccle_proteomics.csv',
    },
}


def ic50_transform(x):
    '''
    Rescale ln(IC50) values to [0, 1] with y = 1 / (1 + IC50^-0.1)
    '''
    return 1 / (1 + np.exp(x) ** (-0.1))


TARGET_TRANSFORMS = {
    'sigmoid': ic50_transform,  # ln(IC50) -> [0, 1]
    'minmax': lambda x : MinMaxScaler((0,1)).fit_transform(x.reshape(-1,1)).squeeze(),
    'none': None,  # values already in [0, 1], e.g. AUDRC
}


def encode_drugs_if_needed(drugs_file, out_dir):
    '''
    Drug files with only drug_name and SMILES are encoded as molecular graphs
    '''
    columns = pd.read_csv(drugs_file, nrows=0).columns
    if 'atomic_features' in columns:
        return drugs_file
    from drugs_encoding import encode_drugs
    smiles_key = [c for c in columns if c.lower() in ['smiles', 'canonicalsmiles']]
    assert smiles_key, f"{drugs_file} must contain a smiles column"
    makedirs(out_dir, exist_ok=True)
    encoded = path.join(out_dir, 'drugs_encoded.csv')
    encode_drugs(drugs_file, encoded, smiles_key=smiles_key[0])
    return encoded


def build_dataset(name, out_dir, response, drugs, rnaseq=None, proteomics=None,
                  target='IC50', transform='sigmoid', response_filter=None):
    '''
    Build and serialize an ER-graph dataset.

    Args:
        response: csv with cell_line_name, drug_name, <target> (and optionally Max conc)
        drugs: csv with drug_name and smiles, or a file already encoded by drugs_encoding.py
        rnaseq: optional csv with cell_line_name, gene_symbol, tpm
        proteomics: optional csv with cell_line_name, uniprot_id, z-score
        transform: transformation of the target values, one of TARGET_TRANSFORMS
    '''
    for f in [response, drugs, rnaseq, proteomics]:
        assert f is None or path.isfile(f), f"Missing file {f} (run download_data.sh, see README.md)"
    drugs = encode_drugs_if_needed(drugs, out_dir)

    cell_line_e = Entity("cell_line", entity_key='cell_line_name')
    drug_e = Entity("drug", drugs,
                    side_info_features=['atomic_features', 'atomic_bonds', 'fingerprints'],
                    side_info_transf_funs={}, entity_key='drug_name')

    # Main task : drug response, rescaled to [0, 1]
    rel_cell_line_drug = Relation(cell_line_e, drug_e, target,
                                  relation_data_path=response,
                                  dtype=np.float32,
                                  filter_query=response_filter,
                                  transform_fun=TARGET_TRANSFORMS[transform])
    print("Drug-cell line relation stats")
    rel_cell_line_drug.print_stats()
    relations = [rel_cell_line_drug]

    # Maximum tested concentration of each pair (used by NxtDRPMC)
    has_max_conc = 'Max conc' in pd.read_csv(response, nrows=0).columns
    if has_max_conc:
        rel_max_conc = Relation(cell_line_e, drug_e, 'Max conc',
                                relation_data_path=response,
                                dtype=np.float32,
                                filter_query=response_filter)

    scaler = MinMaxScaler((0,1))

    # Proteomics
    if proteomics is not None:
        proteomics_e = Entity("protein", entity_key='uniprot_id')
        rel_cell_line_protein = Relation(cell_line_e, proteomics_e, 'z-score',
                                         relation_data_path=proteomics,
                                         dtype=np.float32,
                                         transform_fun=lambda x : scaler.fit_transform(x.reshape(-1,1)).squeeze())
        rel_cell_line_protein.print_stats()
        relations.append(rel_cell_line_protein)

    # RNA-Seq (TPM = 0 is treated as not observed)
    if rnaseq is not None:
        rnaseq_e = Entity("gene", entity_key='gene_symbol')
        rel_cell_line_rnaseq = Relation(cell_line_e, rnaseq_e, 'tpm',
                                        relation_data_path=rnaseq,
                                        dtype=np.float32,
                                        filter_query='tpm > 0',
                                        transform_fun=lambda x :scaler.fit_transform(np.log(x).reshape(-1,1)).squeeze())
        rel_cell_line_rnaseq.print_stats()
        relations.append(rel_cell_line_rnaseq)

    DatasetHandler(relations, serialize_path=out_dir, overwrite=True)

    if has_max_conc:
        rel_max_conc._set_rel_matrix()
        assert rel_max_conc.matrix.shape == rel_cell_line_drug.matrix.shape
        sparse.save_npz(path.join(out_dir, 'relations', 'drug_response_max_conc.npz'),
                        rel_max_conc.matrix)

    with open(path.join(out_dir, 'dataset_info.json'), 'w') as f:
        json.dump({'name': name, 'target': target, 'transform': transform,
                   'response': path.abspath(response), 'response_filter': response_filter,
                   'relations': [r.name for r in relations]}, f, indent=2)
    print(f"Dataset {name} saved in {out_dir}")


def load_dataset_info(dataset_path):
    with open(path.join(dataset_path, 'dataset_info.json')) as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Build the serialized ER-graph datasets: the paper datasets (--dataset) '
                    'or a dataset from your own files (--custom NAME --response ... --drugs ...)')
    parser.add_argument('--dataset', default='all', choices=list(DATASETS.keys()) + ['all'])
    parser.add_argument('--raw_dir', default='./data/raw')
    parser.add_argument('--out_dir', default='./data/datasets',
                        help='Each dataset is saved in <out_dir>/<name>/')
    custom = parser.add_argument_group('custom dataset')
    custom.add_argument('--custom', default=None, metavar='NAME', help='Name of the custom dataset')
    custom.add_argument('--response', help='csv with cell_line_name, drug_name and the target column')
    custom.add_argument('--drugs', help='csv with drug_name and smiles (or encoded by drugs_encoding.py)')
    custom.add_argument('--rnaseq', default=None, help='csv with cell_line_name, gene_symbol, tpm')
    custom.add_argument('--proteomics', default=None, help='csv with cell_line_name, uniprot_id, z-score')
    custom.add_argument('--target', default='IC50', help='Target column of --response (default IC50)')
    custom.add_argument('--transform', default='sigmoid', choices=list(TARGET_TRANSFORMS.keys()),
                        help='sigmoid for ln(IC50), none for values already in [0, 1], minmax otherwise')
    custom.add_argument('--response_filter', default=None,
                        help='Optional pandas query on --response, e.g. "rmse <= 0.3"')
    args = parser.parse_args()

    if args.custom:
        assert args.response and args.drugs, "--custom requires --response and --drugs"
        build_dataset(args.custom, path.join(args.out_dir, args.custom),
                      args.response, args.drugs, args.rnaseq, args.proteomics,
                      args.target, args.transform, args.response_filter)
    else:
        datasets = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
        for dataset in datasets:
            conf = DATASETS[dataset]
            build_dataset(dataset, path.join(args.out_dir, dataset),
                          **{k: path.join(args.raw_dir, conf[k])
                             for k in ['response', 'drugs', 'rnaseq', 'proteomics']},
                          target=conf['target'], transform=conf['transform'],
                          response_filter=conf['response_filter'])
