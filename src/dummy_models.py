#!/usr/bin/env python3
'''
Dummy baselines of the paper, evaluated with the same splits and validation of NxtDRP:
  - DummyDrugAvg (unseen_cell)  : average response of the drug on the training cell lines
  - DummyCellAvg (unseen_drug)  : average response of the cell line on the training drugs
  - DummyLR      (random_split) : linear regression on one-hot drug and cell line identifiers
  - DummyMC      (unseen_drug)  : maximum tested concentration of the drug
'''

import argparse
import os
import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection
from scipy import sparse
from data import TARGET_TRANSFORMS, load_dataset_info
from validation import evaluate, summary_table


def load_response(dataset_path):
    '''
    Drug response of a dataset built by data.py, with the same filters and target transformation
    '''
    info = load_dataset_info(dataset_path)
    df = pd.read_csv(info['response'])
    if info['response_filter'] is not None:
        df = df.query(info['response_filter'])
    df = df.dropna(subset=['drug_name', 'cell_line_name', info['target']])
    df['drug_name'] = df['drug_name'].str.lower()
    df['cell_line_name'] = df['cell_line_name'].astype(str).str.lower()
    df = df.drop_duplicates(['drug_name', 'cell_line_name'], keep='last').reset_index(drop=True)
    transform = TARGET_TRANSFORMS[info['transform']]
    y = df[info['target']].to_numpy(dtype=np.float32)
    df['y'] = transform(y) if transform is not None else y
    if 'Max conc' in df.columns:
        df['max_conc'] = 1 / (1 + df['Max conc'] ** (-0.1))
    return df


def predict(model_name, train, test):
    if model_name == 'DummyDrugAvg':
        avg = train.groupby('drug_name')['y'].mean()
        return test['drug_name'].map(avg).fillna(train['y'].mean()).values
    if model_name == 'DummyCellAvg':
        avg = train.groupby('cell_line_name')['y'].mean()
        return test['cell_line_name'].map(avg).fillna(train['y'].mean()).values
    if model_name == 'DummyMC':
        return test['max_conc'].values
    if model_name == 'DummyLR':
        ids = pd.concat([train, test])
        drugs = {d: i for i, d in enumerate(ids['drug_name'].unique())}
        cells = {c: i + len(drugs) for i, c in enumerate(ids['cell_line_name'].unique())}

        def one_hot(df):
            rows = np.repeat(np.arange(len(df)), 2)
            cols = np.stack([df['drug_name'].map(drugs), df['cell_line_name'].map(cells)], 1).ravel()
            return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(drugs) + len(cells)))

        model = linear_model.LinearRegression()
        model.fit(one_hot(train), train['y'].values)
        return model.predict(one_hot(test))
    raise ValueError(model_name)


# Model -> splitting strategy (as in the paper)
DUMMY_MODELS = {
    'DummyDrugAvg': 'unseen_cell',
    'DummyCellAvg': 'unseen_drug',
    'DummyMC': 'unseen_drug',
    'DummyLR': 'random_split',
}


def run_dummy(model_name, df, n_tests, seed, out_dir):
    cv_type = DUMMY_MODELS[model_name]
    if cv_type == 'random_split':
        cv = model_selection.ShuffleSplit(n_splits=n_tests, test_size=0.1, random_state=seed)
        groups = None
    else:
        cv = model_selection.GroupShuffleSplit(n_splits=n_tests, test_size=0.1, random_state=seed)
        groups = df['cell_line_name'] if cv_type == 'unseen_cell' else df['drug_name']

    os.makedirs(out_dir, exist_ok=True)
    for i, (train_idx, test_idx) in enumerate(cv.split(df, groups=groups)):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        pd.DataFrame({'cell': test['cell_line_name'].values,
                      'drug': test['drug_name'].values,
                      'true_value': test['y'].values,
                      'predicted_value': predict(model_name, train, test)}
                     ).to_csv(os.path.join(out_dir, f"split_{i + 1:02d}.csv"), index=False)
    return evaluate(out_dir, save_metrics=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dummy baselines (DummyDrugAvg, DummyCellAvg, DummyLR, DummyMC)')
    parser.add_argument('--dataset', default='gdsc',
                        help='Dataset built by src/data.py: gdsc, gdsc_auc, ccle or a custom one')
    parser.add_argument('--datasets_dir', default='data/datasets')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--n_tests', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1956)
    args = parser.parse_args()

    df = load_response(os.path.join(args.datasets_dir, args.dataset))
    perfs = {}
    for model_name, cv_type in DUMMY_MODELS.items():
        if model_name == 'DummyMC' and 'max_conc' not in df.columns:
            continue
        run = f"{args.dataset}_{model_name}_{cv_type}"
        perfs[run] = run_dummy(model_name, df, args.n_tests, args.seed,
                               os.path.join(args.results_dir, run))
    print(summary_table(perfs).to_string(index=False))
