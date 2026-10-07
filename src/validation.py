#!/usr/bin/env python3
'''
Validation pipeline for Drug Response Predictors (from github.com/codicef/DRPValidation).

Predictions are csv files with the columns: cell, drug, true_value, predicted_value
(one file per train/test split). The metrics are computed with three Aggregation Strategies:
  - global      : over the whole test set
  - fixed_drug  : per drug, then averaged (ability to rank cell lines, key for unseen cell lines)
  - fixed_cell  : per cell line, then averaged (ability to rank drugs, key for unseen drugs)
'''

import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import spearmanr, pearsonr


DIGITS = 3

METRICS = {'RMSE': root_mean_squared_error,
           'R2': r2_score,
           'Spearman': lambda x, y: spearmanr(x, y)[0],
           'Pearson': lambda x, y: pearsonr(x, y)[0]}

AGGREGATIONS = ['global', 'fixed_drug', 'fixed_cell']


def split_metrics(df):
    '''
    Metrics of a single train/test split, for each aggregation strategy
    '''
    out = {}
    y_true, y_hat = df['true_value'].values, df['predicted_value'].values
    out['global'] = {m: f(y_true, y_hat) for m, f in METRICS.items()}

    for aggr, key in [('fixed_drug', 'drug'), ('fixed_cell', 'cell')]:
        per_group = {m: [] for m in METRICS}
        for _, df_g in df.groupby(key):
            if len(df_g) < 2:
                continue
            for m, f in METRICS.items():
                per_group[m].append(f(df_g['true_value'].values, df_g['predicted_value'].values))
        out[aggr] = {m: np.nanmean(v) if len(v) else np.nan for m, v in per_group.items()}
    return out


def evaluate(path, save_metrics=False, verbose=True):
    '''
    Evaluate a prediction csv file or a directory of csv files (one per split)

    Returns:
        dict {aggregation: {metric: [value of each split]}}
    '''
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv'))
    else:
        files = [path]
    assert len(files) > 0, f"No prediction csv found in {path}"

    perf = {aggr: {m: [] for m in METRICS} for aggr in AGGREGATIONS}
    for f_path in files:
        res = split_metrics(pd.read_csv(f_path))
        for aggr in AGGREGATIONS:
            for m in METRICS:
                perf[aggr][m].append(float(res[aggr][m]))

    if verbose:
        print(f"Metrics for {path}")
        print(f"Number of splits: {len(files)}")
        for aggr in AGGREGATIONS:
            print(f"{aggr}:")
            for m in METRICS:
                print(f"\t{m}: {round(np.nanmean(perf[aggr][m]), DIGITS)}, "
                      f"std: {round(np.nanstd(perf[aggr][m]), DIGITS)}")
        print()

    if save_metrics:
        save_path = os.path.join(path, 'metrics.json') if os.path.isdir(path) \
            else os.path.splitext(path)[0] + '_metrics.json'
        with open(save_path, 'w') as f:
            json.dump(perf, f)
        print(f"Metrics saved in {save_path}")
    return perf


def summary_table(perfs):
    '''
    perfs: {run_name: output of evaluate} -> DataFrame with mean and std of each metric
    '''
    rows = []
    for run, perf in perfs.items():
        for aggr in AGGREGATIONS:
            row = {'run': run, 'aggregation': aggr, 'n_splits': len(perf[aggr]['RMSE'])}
            for m in METRICS:
                row[m] = round(np.nanmean(perf[aggr][m]), DIGITS)
                row[m + '_std'] = round(np.nanstd(perf[aggr][m]), DIGITS)
            rows.append(row)
    return pd.DataFrame(rows)


def convert(path, conv_path):
    '''
    Convert the legacy pickle predictions ((train), (x_test, y_test, y_hat_test)) into csv files
    '''
    os.makedirs(conv_path, exist_ok=True)
    for f_name in sorted(os.listdir(path)):
        with open(os.path.join(path, f_name), 'rb') as f:
            obj = pickle.load(f)
        test_p = obj[1]
        if len(test_p[0]) > 2:
            cells, drugs = list(zip(*test_p[0]))
        else:
            cells, drugs = test_p[0]
        df = pd.DataFrame({'cell': cells, 'drug': drugs,
                           'true_value': np.asarray(test_p[1]).squeeze(),
                           'predicted_value': np.asarray(test_p[2]).squeeze()})
        df.to_csv(os.path.join(conv_path, f_name + '.csv'), index=False)
    print(f"Converted predictions saved in {conv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DRP predictions with Global, Fixed-Drug and Fixed-Cell Line aggregation')
    sub = parser.add_subparsers(dest='command', required=True)

    p_eval = sub.add_parser('evaluate', help='Compute metrics of prediction csv files')
    p_eval.add_argument('paths', nargs='+', help='csv files or directories with one csv per split')
    p_eval.add_argument('--save_metrics', action='store_true', help='Save the per-split metrics as json')
    p_eval.add_argument('--summary', default=None, help='Save a summary table (csv) of all the paths')

    p_conv = sub.add_parser('convert', help='Convert legacy pickle predictions into csv files')
    p_conv.add_argument('path', help='Directory with the pickle files')
    p_conv.add_argument('conv_path', help='Output directory')

    args = parser.parse_args()
    if args.command == 'evaluate':
        perfs = {p: evaluate(p, args.save_metrics) for p in args.paths}
        if args.summary:
            summary_table(perfs).to_csv(args.summary, index=False)
            print(f"Summary saved in {args.summary}")
    else:
        convert(args.path, args.conv_path)
