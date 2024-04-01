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
                 ):

        self.serialize_path = serialize_path
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
                rel.save(path.join(serialize_path, 'relations'))

                if rel.name == 'cell_line-drug':
                    get_max_conc_data(rel, './data/raw/relations/ccle/concat/concat_drug_response.csv',
                                      path.join(serialize_path, 'relations', 'drug_response_max_conc.npz'))
                # save entities
                #
                rel._set_rel_matrix()
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
        kf = GroupShuffleSplit(n_splits=n_splits, test_size=0.1)#, random_state=1991)
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
        kf = ShuffleSplit(n_splits=n_splits,  random_state=1956, test_size=test_size)


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
                    train_idx, test_idx = list(set(range(len(active_indices)))-set(fixed_test_indices)), fixed_test_indices
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
    def load_serialized(serialized_path, load_side_info=True):
        assert path.isdir(path.join(serialized_path, 'relations')) and path.isdir(
            path.join(serialized_path, 'entities'))

        ent_d = {}
        for ent_file in listdir(path.join(serialized_path, 'entities')):
            entity = Entity.load_saved(path.join(serialized_path, 'entities', ent_file),
                                       load_side_info=load_side_info)
            ent_d[entity.name] = entity


        relations = []
        for rel_file  in list(listdir(path.join(serialized_path, 'relations'))):
            if '-' not in rel_file:
                continue
            if 'idx' in rel_file:
                continue
            # rel_file_name = str(rel_file)
            # rel_file_name.replace('.npy', '')
            ee = rel_file.replace('.npz', '').split('-')
            e0, e1 = ee[0], ee[1]
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
            df = df.dropna()
            print(f"Relation file loaded, columns : {df.columns}")
            if entity_0.key == entity_1.key:
                e0_key = entity_0.key + '_1'
                e1_key = entity_1.key + '_2'
            else:
                e0_key, e1_key = entity_0.key, entity_1.key
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




def get_max_conc_data(relation, file_path, out_path):
    df = pd.read_csv(file_path)
    df = df[['cell_line_name', 'drug_name', 'Max conc']]

    #lowercase
    df['cell_line_name'] = df['cell_line_name'].str.lower()
    df['drug_name'] = df['drug_name'].str.lower()

    e0_keys = relation.e0_keys
    e1_keys = relation.e1_keys

    e0_indices = [relation.entity_0.e_idx[e] for e in e0_keys]
    e1_indices = [relation.entity_1.e_idx[e] for e in e1_keys]

    # get max conc for rows correspondi to e0_keys=cell_line_name, and cols e1_keys=drug_name, they represent pairs

    df.set_index(['cell_line_name', 'drug_name'], inplace=True)
    df = df.loc[list(zip(e0_keys, e1_keys))]
    df = df.reset_index()
    max_conc = df['Max conc'].to_numpy()

    # save in sparse matrix
    matrix = sparse.coo_matrix((max_conc, (e0_indices, e1_indices)),
                                 shape=(len(relation.entity_0.idx_e), len(relation.entity_1.idx_e)),
                                    dtype=np.float32)
    sparse.save_npz(out_path, matrix)



if __name__ == '__main__':
    print("Test dataset classes / Generate datasets")

    if not path.isdir('./data/datasets'):
        makedirs('./data/datasets')

    cell_line_e = Entity("cell_line", entity_key='cell_line_name')


    # # # Drug compounds data
    drug_e = Entity("drug", "./data/raw/entities_info/drugs_v6.csv",
                    side_info_features=['atomic_features', 'atomic_bonds', 'fingerprints'],
                    side_info_transf_funs={}, entity_key='drug_name'
                    )

    scaler = MinMaxScaler((0,1))
    # scaler = StandardScaler()

    rel_cell_line_drug_auc = Relation(cell_line_e, drug_e, 'auc',
                                  relation_data_path='./data/raw/relations/gdsc_drug_cellline_v6_nd.csv',
                                  dtype=np.float32,
                                  transform_fun=lambda x : scaler.fit_transform(x.reshape(-1,1)).squeeze())
                                  #transform_fun=lambda x : 1 / (1 + (np.exp(x)**(-0.1))))# , filter_query='rmse < 0.1')

    rel_cell_line_drug = Relation(cell_line_e, drug_e, 'IC50',
                                  relation_data_path='./data/raw/relations/gdsc_drug_cellline_v6_nd.csv',
                                  dtype=np.float32,
                                  transform_fun=lambda x : 1 / (1 + (np.exp(x)**(-0.1))))# , filter_query='rmse < 0.1')



    #Rna seq data
    rnaseq_e = Entity("gene", entity_key='gene_symbol')

    # with open("./data/raw/relations/all_cosmic_census.txt", 'r') as f:
    #      filter_list = f.read().splitlines()
    rel_cell_line_rnaseq = Relation(cell_line_e, rnaseq_e, 'tpm',
                                    relation_data_path='./data/raw/relations/rnaseq_tpm_cellline_v6_top500.csv',
                                    dtype=np.float32,
                                    transform_fun=lambda x :scaler.fit_transform(np.log(x).reshape(-1,1)).squeeze())
    rel_cell_line_rnaseq.print_stats()



    # # Load genomics mut data
    # rel_cell_line_mut = Relation(cell_line_e, rnaseq_e, 'value',
    #                                 relation_data_path='./data/raw/relations/cell_gene_mutations.csv',
    #                                 dtype=np.int16,
    #                                 transform_fun=lambda x : scaler.fit_transform(x.reshape(-1,1)).squeeze())
    # rel_cell_line_mut.print_stats()

    

    # # # Proteomics data
    proteomics_e = Entity("protein", entity_key='uniprot_id')

    rel_cell_line_protein = Relation(cell_line_e, proteomics_e, 'z-score',
                                     relation_data_path='./data/raw/relations/protein_zscore_cellline_v6_l.csv',
                                     dtype=np.float32,
                                     transform_fun=lambda x : scaler.fit_transform(x.reshape(-1,1)).squeeze())

    rel_cell_line_protein.print_stats()




    # # # Prot gene relation
    # # rel_prot_rnaseq = Relation(proteomics_e, rnaseq_e, 'binary',
    # #                             relation_data_path='./data/raw/relations/gene_proteins.csv',
    # #                                 dtype=np.int16)
    # # rel_prot_rnaseq.print_stats()



    # # # Chemchemrelation
    rel_chem_chem = Relation(drug_e, drug_e, 'combined_score',
                             relation_data_path='./data/raw/relations/stitch_chemical_chemical.csv',
                             dtype=np.float32,
                             transform_fun=lambda x : scaler.fit_transform(x.reshape(-1,1)).squeeze(),

                             )


    ds_mf = DatasetHandler([rel_cell_line_drug], serialize_path='./data/datasets/base_gnn/',
                           overwrite=True)

    ds_chem = DatasetHandler([rel_cell_line_drug, rel_chem_chem],
                         serialize_path='./data/datasets/base_chem_gnn/',
                           overwrite=True)

    ds_rnaseq = DatasetHandler([rel_cell_line_drug, rel_cell_line_rnaseq],
                        serialize_path='./data/datasets/base_rnaseq_gnn/',
                               overwrite=True)


    ds_prot = DatasetHandler([rel_cell_line_drug, rel_cell_line_protein],
                        serialize_path='./data/datasets/base_prot_gnn/',
                          overwrite=True)

    ds_prot_rna = DatasetHandler([rel_cell_line_drug, rel_cell_line_protein, rel_cell_line_rnaseq],
                         serialize_path='./data/datasets/base_prot_rnaseq_gnn/',
                           overwrite=True)

    ds_prot_rna_chem = DatasetHandler([rel_cell_line_drug, rel_cell_line_protein, rel_chem_chem,
                                       rel_cell_line_rnaseq],
                         serialize_path='./data/datasets/base_prot_rnaseq_chem_gnn/',
                           overwrite=True)




    ds_mf = DatasetHandler([rel_cell_line_drug_auc], serialize_path='./data/datasets/base_auc/',
                           overwrite=True)

    ds_chem = DatasetHandler([rel_cell_line_drug_auc, rel_chem_chem],
                         serialize_path='./data/datasets/base_chem_auc/',
                           overwrite=True)

    ds_rnaseq = DatasetHandler([rel_cell_line_drug_auc, rel_cell_line_rnaseq],
                        serialize_path='./data/datasets/base_rnaseq_auc/',
                               overwrite=True)


    ds_prot = DatasetHandler([rel_cell_line_drug_auc, rel_cell_line_protein],
                        serialize_path='./data/datasets/base_prot_auc/',
                          overwrite=True)

    ds_prot_rna = DatasetHandler([rel_cell_line_drug_auc, rel_cell_line_protein, rel_cell_line_rnaseq],
                         serialize_path='./data/datasets/base_prot_rnaseq_auc/',
                           overwrite=True)

    ds_prot_rna_chem = DatasetHandler([rel_cell_line_drug_auc, rel_cell_line_protein, rel_chem_chem,
                                       rel_cell_line_rnaseq],
                         serialize_path='./data/datasets/base_prot_rnaseq_chem_auc/',
                           overwrite=True)
