#!/usr/bin/env python3

from data import DatasetHandler
from models import NxtDRPMC, NxtDRP
from evaluate import evaluate_regression
from validation import evaluate, summary_table
from utils import compute_similarities, FocalLoss, log_message
import torch
from NXTfusion import NXLosses
from NXTfusion.NXmultiRelSide import NNwrapper
from NXTfusion.NXFeaturesConstruction import buildPytorchFeats
import seaborn as sns
import os
import random
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
import traceback

N_CV_SPLITS = 4
N_TRIALS = 80

def randomized_test(ds, n_tests, model_class, losses_dict, target_relation,
                    device='cuda',
                    cv_type='cell',
                    fixed_hyperparameters=None,
                    test_indices=None,
                    loaded_model=None,
                    run_name='',
                    results_dir=None):
    print(f"\n\nRandomized test with {n_tests} random splits\nSplitting Strategy : {cv_type}\nModel : {model_class.__name__}\nDevice : {device}\nFixed Hyperparameters : {fixed_hyperparameters is not None}")


    ds_model_name = [e.name for e in ds.entities]
    ds_model_name = "_".join(ds_model_name)

    log_file_name = datetime.now().strftime("%Y_%m_%d_%H-%M-%S") + '_' + run_name + '_' + \
        ds_model_name + '_' + cv_type + '_hpopt_' + str(fixed_hyperparameters is None) + '.log'


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
                                                 cv_type, model_class, cv_splits=N_CV_SPLITS,
                                                 device=device)
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
            if loaded_model is None:
                model = model_class(er, 'simple_test', main_emb_size=hyperparameters['emb_size'],
                                    dropout=hyperparameters['dropout'],
                                    gnn_size=hyperparameters['gnn_size'],
                                    out_emb_size=hyperparameters['out_emb_size'],
                                    device=device,
                                    side_info_file=os.path.join(ds.serialize_path, 'entities', 'drug.csv'))
            else:
                model = loaded_model


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

            if test_indices is not None:
                model.save(os.path.join('./log/', 'model_' + log_file_name + '.pkl'))

            log_message(log_file_name, f"Test fold : {test_i + 1}\nTrain Perf {train_perf}")

            x_test, y_test, corr = buildPytorchFeats(test_idx, nx_target['domain1'],
                                                     nx_target['domain2'])

            y_hat_test = wrapper.predict(er, x_test, target_relation.name, target_relation.name,
                                         sidex1=side_info_d[nx_target['domain1'].name],
                                         sidex2=side_info_d[nx_target['domain2'].name],
                                         batch_size=hyperparameters['batch_size'])

            test_perf = evaluate_regression(y_hat_test, y_test, plot=False)


            log_message(log_file_name, f"Test fold : {test_i + 1}\n*************\nTest Perf {test_perf}\n")

            if results_dir is not None:
                save_predictions(ds, x_test, y_test, y_hat_test,
                                 os.path.join(results_dir, f"split_{test_i + 1:02d}.csv"))


        except Exception as e:
            log_message(log_file_name, f"Error : {str(e)}\n{traceback.format_exc()}")
            raise

    if results_dir is not None:
        perf = evaluate(results_dir, save_metrics=True, verbose=False)
        log_message(log_file_name, "Test performance averaged over the splits\n" +
                    summary_table({run_name: perf}).to_string(index=False))


def save_predictions(ds, x, y_true, y_hat, out_file):
    '''
    Save the predictions of a split in the format read by validation.py
    '''
    cells, drugs = zip(*x)
    pd.DataFrame({'cell': ds.entities_dict['cell_line'].idx_e[list(cells)],
                  'drug': ds.entities_dict['drug'].idx_e[list(drugs)],
                  'true_value': np.asarray(y_true).squeeze(),
                  'predicted_value': np.asarray(y_hat).squeeze()}).to_csv(out_file, index=False)









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
                                device=device,
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
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed : {e}")
        return 0
    return perf[0]['pearson']


OMICS_RELATIONS = {
    'none': ['cell_line-drug'],
    'pr': ['cell_line-drug', 'cell_line-protein'],
    'ex': ['cell_line-drug', 'cell_line-gene'],
    'pr_ex': ['cell_line-drug', 'cell_line-protein', 'cell_line-gene'],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_losses_dict():
    loss_f = NXLosses.LossWrapper(torch.nn.L1Loss(reduction='mean'),
                                  type='regression', ignore_index=0)

    return {
        'cell_line-drug': loss_f,
        'drug-drug': loss_f,
        'drug-gene': loss_f,
        'cell_line-gene': loss_f,
        'cell_line-protein': loss_f,
    }


def main(model_class, cv_type, dataset_path, device='cuda', default_hp_path=None,
         n_tests=1, test_indices=None, load_model_path=None, omics='pr_ex', seed=1956,
         results_dir=None):

    losses_dict = get_losses_dict()

    if default_hp_path:
        with open(default_hp_path, 'r') as f:
            fixed_hyperparameters_all = json.load(f)
        fixed_hp = fixed_hyperparameters_all[cv_type]
    else:
        fixed_hp = None

    assert os.path.isdir(dataset_path), f"Dataset {dataset_path} not found, run src/data.py first"
    ds = DatasetHandler.load_serialized(dataset_path, load_side_info=False,
                                        relations_to_load=OMICS_RELATIONS[omics])
    ds.seed = seed

    loaded_model = None
    if load_model_path is not None:
        print(f"\n==> Loading model from {load_model_path}")
        with open(load_model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print("Model loaded successfully!")

    print(f"\n==> Running randomized evaluation on {dataset_path} ({n_tests} splits)")
    randomized_test(ds=ds,
                    n_tests=n_tests,
                    model_class=model_class,
                    losses_dict=losses_dict,
                    target_relation=ds.rel_dict['cell_line-drug'],
                    device=device,
                    cv_type=cv_type,
                    fixed_hyperparameters=fixed_hp,
                    test_indices=test_indices,
                    loaded_model=loaded_model,
                    run_name=os.path.basename(os.path.normpath(results_dir)) if results_dir
                    else model_class.__name__,
                    results_dir=results_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run NxtDRP randomized train/test evaluation')
    parser.add_argument('--dataset', type=str, default='gdsc', choices=['gdsc', 'gdsc_auc', 'ccle'])
    parser.add_argument('--datasets_dir', type=str, default='data/datasets',
                        help='Folder containing the datasets built by src/data.py')
    parser.add_argument('--model', type=str, default='NxtDRP', choices=['NxtDRP', 'NxtDRPMC'])
    parser.add_argument('--omics', type=str, default='pr_ex', choices=list(OMICS_RELATIONS.keys()),
                        help='Cell line omics added to the ER graph: none (MT), pr (MT+PR), ex (MT+EX), pr_ex (MT+PR+EX)')
    parser.add_argument('--default_hp_path', type=str, default='data/hyperparameters/default_hp.json',
                        help='Path to JSON with fixed hyperparameters')
    parser.add_argument('--optimize_hp', action='store_true',
                        help='Optimize hyperparameters with optuna instead of using --default_hp_path')
    parser.add_argument('--test_indices_path', type=str, default=None,
                        help='Path to pickle file with test indices')
    parser.add_argument('--cv_type', type=str, default='random_split', choices=['random_split', 'unseen_cell', 'unseen_drug'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_tests', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1956)
    parser.add_argument('--load_model_path', type=str, default=None,
                        help='Path to a saved model to load instead of training a new one')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Predictions are saved in <results_dir>/<dataset>_<model>_<omics>_<cv_type>/')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = 'cpu'

    set_seed(args.seed)

    model_class = NxtDRP if args.model == 'NxtDRP' else NxtDRPMC
    map_cv = {'random_split': 'cell', 'unseen_cell': 'row', 'unseen_drug': 'col'}
    cv_type = map_cv[args.cv_type]

    # Check if test indices are provided
    if args.test_indices_path is not None:
        assert os.path.isfile(args.test_indices_path), f"Test indices file {args.test_indices_path} not found"
        with open(args.test_indices_path, 'rb') as f:
            test_indices = pickle.load(f)
        n_tests = 1
    else:
        test_indices = None
        n_tests = args.n_tests

    run_dir = os.path.join(args.results_dir,
                           f"{args.dataset}_{args.model}_{args.omics}_{args.cv_type}")
    os.makedirs(run_dir, exist_ok=True)
    for f in os.listdir(run_dir):  # remove the splits of previous runs
        if f.startswith('split_'):
            os.remove(os.path.join(run_dir, f))
    os.makedirs('./log/', exist_ok=True)

    main(model_class=model_class,
         cv_type=cv_type,
         dataset_path=os.path.join(args.datasets_dir, args.dataset),
         device=args.device,
         default_hp_path=None if args.optimize_hp else args.default_hp_path,
         n_tests=n_tests,
         test_indices=test_indices,
         load_model_path=args.load_model_path,
         omics=args.omics,
         seed=args.seed,
         results_dir=run_dir)
