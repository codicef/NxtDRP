#!/usr/bin/env python3

from scipy.special import hyp0f1
from data import DatasetHandler
from models import NxtDRPMC, NxtDRP
from evaluate import evaluate_regression
from utils import compute_similarities, FocalLoss, log_message
import torch
from NXTfusion import NXLosses
from NXTfusion.NXmultiRelSide import NNwrapper
from NXTfusion.NXFeaturesConstruction import buildPytorchFeats
import seaborn as sns
from matplotlib import cbook, pyplot as plt
import os
import logging
import optuna
from datetime import datetime
from os import path
import numpy as np
import pickle
import pandas as pd
import argparse
import json
import sys

N_CV_SPLITS = 4
N_TRIALS = 80


def randomized_test(ds, n_tests, model_class, losses_dict, target_relation,
                    device='cuda',
                    cv_type='cell',
                    fixed_hyperparameters=None,
                    test_indices=None):
    print(f"\n\nRandomized test with {n_tests} random splits\nSplitting Strategy : {cv_type}\nModel : {model_class.__name__}\nDevice : {device}\nFixed Hyperparameters : {fixed_hyperparameters is not None}")


    ds_model_name = [e.name for e in ds.entities]
    ds_model_name = "_".join(ds_model_name)

    log_file_name = datetime.now().strftime("%m_%d_%Y_%H:%M") + '_' + \
        ds_model_name + '_' + cv_type +'_hp_' + str(fixed_hyperparameters is None) + '.log'


    if fixed_hyperparameters is None:
        logger = logging.getLogger()

        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler(os.path.join('./log/','optuna_' + log_file_name), mode="w"))

        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    else:
        log_message(log_file_name, f"Fixed Hyperparameters : {fixed_hyperparameters}")



    for test_i in range(n_tests):
        log_message(log_file_name, f"Randomization split n={test_i + 1}")

        er, train_idx, test_idx, nx_target, side_info_d = \
            ds.get_cell_cv_folds(target_relation,
                                         losses_dict,
                                         test_i,
                                         n_tests,
                                         cv_type=cv_type,
                                 split_type='test',
                                 fixed_test_indices=test_indices)

        if fixed_hyperparameters is None:
            print("Optimizing hyperpameters")
            fun_obj = lambda trial : objective_optuna(trial, ds, losses_dict,
                                                 cv_type, model_class, cv_splits=N_CV_SPLITS)
            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
            study.optimize(fun_obj, n_trials=N_TRIALS, n_jobs=1)
            trial = study.best_trial


            par = ""
            for key, value in trial.params.items():
                par += "\n    {}: {}".format(key, value)
            log_message(log_file_name, f"Best trial RP:{trial.value} \n Params: {par}")


            hyperparameters = trial.params
            hyperparameters['dropout'] = 0
        else:
            hyperparameters = fixed_hyperparameters
        print(f"Hyperparameters : {hyperparameters}")


        try:
            x_train, y_train, corr_ = buildPytorchFeats(train_idx, nx_target['domain1'],
                                                    nx_target['domain2'])

            # Build model
            model = model_class(er, 'simple_test', main_emb_size=hyperparameters['emb_size'],
                                dropout=hyperparameters['dropout'],
                                gnn_size=hyperparameters['gnn_size'],
                                out_emb_size=hyperparameters['out_emb_size'],
                                device=device,
                                side_info_file=os.path.join(ds.serialize_path, 'entities', 'drug.csv'))
            wrapper = NNwrapper(model, dev=device, ignore_index=-1)
            wrapper.fit(er, LOG=False, epochs=hyperparameters['epochs'],
                        weight_decay=hyperparameters['weight_decay'],
                        batch_size=hyperparameters['batch_size'])

            y_hat_train = wrapper.predict(er, x_train, target_relation.name,
                                          target_relation.name,
                                          sidex1=side_info_d[nx_target['domain1'].name],
                                          sidex2=side_info_d[nx_target['domain2'].name],
                                          batch_size=hyperparameters['batch_size'])


            train_perf = evaluate_regression(y_hat_train, y_train)
            log_message(log_file_name, f"Test fold : {test_i + 1}\nTrain Perf {train_perf}")

            x_test, y_test, corr = buildPytorchFeats(test_idx, nx_target['domain1'],
                                                     nx_target['domain2'])

            y_hat_test = wrapper.predict(er, x_test, target_relation.name, target_relation.name,
                                         sidex1=side_info_d[nx_target['domain1'].name],
                                         sidex2=side_info_d[nx_target['domain2'].name],
                                         batch_size=hyperparameters['batch_size'])

            test_perf = evaluate_regression(y_hat_test, y_test, plot=False)


            log_message(log_file_name, f"Test fold : {test_i + 1}\n*************\nTest Perf {test_perf}\n")

            out_pred = ((x_train, y_train, y_hat_train), (x_test, y_test, y_hat_test))
            with open('./log/preds/' + log_file_name + '_' + str(test_i), 'wb') as f:
                pickle.dump(out_pred, f)


        except Exception as e:
            log_message(log_file_name, f"Error : {str(e)}\n")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            log_message(log_file_name, str((exc_type, fname, exc_tb.tb_lineno)))









def cross_validation(ds, target_relation, losses_dict, model_class,
                     device='cuda', n_splits=5,
                     hyperparameters={}, cv_type='cell', trial=None):
    assert n_splits > 1, "n_splits must be greater than 1"
    perf_cv = {}
    # hyperparameters['dropout'] = 0
    print(f"Cross validating with the following hyperpameters : {hyperparameters} on n={n_splits} splits")
    y_hats = []
    y_trues = []

    for cv_fold in range(n_splits):
        print(f"Starting cv fold {cv_fold} ...")
        # Prepare data
        er, train_idx, test_idx, nx_target, side_info_d = ds.get_cell_cv_folds(target_relation,
                                                                               losses_dict,
                                                                               cv_fold,
                                                                               n_splits,
                                                                               cv_type=cv_type)
        x_train, y_train, _ = buildPytorchFeats(train_idx, nx_target['domain1'],
                                                nx_target['domain2'])

        # Build model
        model = model_class(er, 'simple_test', main_emb_size=hyperparameters['emb_size'],
                            dropout=hyperparameters['dropout'],
                            gnn_size=hyperparameters['gnn_size'],
                            out_emb_size=hyperparameters['out_emb_size'],
                            device=device,
                            side_info_file=os.path.join(ds.serialize_path, 'entities', 'drug.csv'))
        wrapper = NNwrapper(model, dev=device, ignore_index=-1)
        wrapper.fit(er, LOG=False, epochs=hyperparameters['epochs'],
                    weight_decay=hyperparameters['weight_decay'],
                    batch_size=max(6, hyperparameters['batch_size']))

        y_hat_train = wrapper.predict(er, x_train, target_relation.name,
                                      target_relation.name,
                                      sidex1=side_info_d[nx_target['domain1'].name],
                                      sidex2=side_info_d[nx_target['domain2'].name],
                                      batch_size=hyperparameters['batch_size'])

        train_perf = evaluate_regression(y_hat_train, y_train)
        print(f"Cv fold : {cv_fold + 1}\nTrain Perf {train_perf}")

        x_test, y_valid, corr = buildPytorchFeats(test_idx, nx_target['domain1'],
                                                 nx_target['domain2'])

        y_hat_valid = wrapper.predict(er, x_test, target_relation.name, target_relation.name,
                                      sidex1=side_info_d[nx_target['domain1'].name],
                                      sidex2=side_info_d[nx_target['domain2'].name],
                                      batch_size=max(6, hyperparameters['batch_size']))

        valid_perf = evaluate_regression(y_hat_valid, y_valid, plot=False)
        print(f"Cv fold : {cv_fold + 1}\nTest Perf {valid_perf}")
        y_hats = y_hats + list(y_hat_valid)
        y_trues = y_trues + list(y_valid)
        cumul_test_perf = evaluate_regression(y_hats, y_trues, plot=False)
        print(f"Cv fold : {cv_fold + 1}\nTest Perf {cumul_test_perf}")
        if trial is not None:
            trial.report(valid_perf['pearson'], cv_fold)
            if trial.should_prune():
                print("Pruning")
                raise optuna.TrialPruned()
            else:
                print("Keep going")

        for k in valid_perf.keys():
            perf_cv[k] = perf_cv.get(k, 0) +  valid_perf[k]



    for k in perf_cv.keys():
        perf_cv[k] /= n_splits
    print(f"Cross validation performaces : {perf_cv}")
    return perf_cv, train_perf



def objective_optuna(trial, ds, losses_dict, cv_type, model_class, hp=None, cv_splits=2, device='cuda'):



    try:
        perf = cross_validation(ds, ds.rel_dict['cell_line-drug'], losses_dict,  model_class,
                                deivice=device,
                                hyperparameters={'epochs': trial.suggest_int('epochs', 100, 250),
                                    'weight_decay': trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
                                    'emb_size': trial.suggest_int('emb_size', 30, 80),
                                    'batch_size': trial.suggest_int('batch_size', 7, 15),
                                    'dropout': trial.suggest_float('dropout', 0.2, 0.6),
                                    'gnn_size': trial.suggest_int('gnn_size', 80, 180),
                                    'out_emb_size': trial.suggest_int('out_emb_size', 20, 50)},

                                n_splits=cv_splits,
                                cv_type=cv_type,
                                trial=trial)
    except:
        return 0
    return perf[0]['pearson']




def main(model_class, n_tests, cv_type, device='cuda', default_hp=None):
    # dataset, load serialized data
    loss_f = NXLosses.LossWrapper(torch.nn.L1Loss(reduction='mean'),
                                  type='regression', ignore_index=0)
                                  
                                  
    losses_dict = {'cell_line-drug': loss_f,
                   'drug-drug': loss_f,
                   'drug-gene': loss_f,
                   'cell_line-gene': loss_f,
                   'cell_line-protein': loss_f,
                   }


    datasets = ['./data/datasets/base_gnn/',
                './data/datasets/base_rnaseq_gnn/',
                './data/datasets/base_prot_gnn/',
                './data/datasets/base_prot_rnaseq_gnn/',
                ]
    for ds in datasets:
        assert os.path.isdir(ds), f"Dataset {ds} not found, please check the path or generate the dataset with data.py"

    if default_hp:
        hp = json.load(open('data/hyperparameters/default_hp.json', 'r'))
    else:
        hp = None

    print(f"Testing type stratification : {cv_type}")

    for d_i, ds_name in enumerate(datasets):
        ds = DatasetHandler.load_serialized(ds_name, load_side_info=False)

        randomized_test(ds, n_tests,
                        model_class,
                        target_relation=ds.rel_dict['cell_line-drug'],
                        losses_dict=losses_dict, fixed_hyperparameters=hp[cv_type],
                        cv_type=cv_type, device=device)


if __name__ == '__main__':
    # Argparse simple setup
    parser = argparse.ArgumentParser(description='Run randomized validation')
    parser.add_argument('--model', type=str, required=False, default='NxtDRP',
                        help='Model to use for the test', choices=['NxtDRP', 'NxtDRPMC'])
    parser.add_argument('--n_tests', type=int, required=False, default=40,
                        help='Number of random tests to perform')
    parser.add_argument('--cv_type', type=str, required=False, default='random_split',
                        help='Type of splitting strategy to use for the test',
                        choices=['random_split', 'useen_cell', 'unseen_drug'])
    parser.add_argument('--default_hp', type=bool, required=False, default=True,
                        help='Use default hyperparameters or optimize them')
    parser.add_argument('--device', type=str, required=False, default='cuda',
                        help='Device to use for the test')

    args = parser.parse_args()

    device = args.device
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("Cuda not available, switching to cpu")
            device = 'cpu'

    model = args.model
    if model == 'NxtDRP':
        model_class = NxtDRP
    else:
        model_class = NxtDRPMC

    cv_type = args.cv_type
    map_cv = {'random_split': 'cell',
              'unseen_cell': 'row',
              'unseen_drug': 'col'}
    cv_type = map_cv[cv_type]


    # Check dirs
    if not os.path.isdir('./log/preds/'):
        os.makedirs('./log/preds/')

    main(model_class,
         n_tests=args.n_tests,
         cv_type=cv_type,
         device=device,
         default_hp=args.default_hp)
