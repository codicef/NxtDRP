#!/usr/bin/env python3
'''
Predict the drug response of cell line - drug pairs with a model trained by train.py.

Cell lines must be part of the dataset the model was trained on (to add new cell lines,
add their omics to the raw files, rebuild the dataset and retrain). Drugs can be any
drug of the dataset or new compounds given as SMILES (--new_drugs).
'''

import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from drugs_encoding import get_atomic_features


def score_to_ln_ic50(y):
    '''
    Inverse of y = 1 / (1 + IC50^-0.1)
    '''
    y = np.clip(y, 1e-6, 1 - 1e-6)
    return -10 * np.log(1 / y - 1)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = ckpt['model']
    model.device = device
    model.batches = {}
    model.to(device)
    model.eval()
    return ckpt, model


def add_new_drugs(model, drugs, new_drugs_file):
    '''
    Append the molecular graphs of new compounds to the model, returns the updated drug list
    '''
    df = pd.read_csv(new_drugs_file)
    drugs = list(drugs)
    for _, row in df.iterrows():
        name = str(row['drug_name']).lower()
        assert name not in drugs, f"Drug {name} is already in the dataset"
        x, bonds, _ = get_atomic_features(row['smiles'])
        assert x is not None and len(bonds) > 0, f"Invalid SMILES for {name}"
        model.graphs.append(pyg.data.Data(
            torch.tensor(x, dtype=torch.float, device=model.device),
            torch.tensor(bonds, dtype=torch.long, device=model.device).transpose(1, 0),
            device=model.device))  # same attributes of the graphs built by the model
        drugs.append(name)
    return drugs


def get_pairs(args, cell_lines, drugs):
    if args.pairs:
        df = pd.read_csv(args.pairs)
        pairs = list(zip(df['cell_line_name'].astype(str).str.lower(), df['drug_name'].str.lower()))
    else:
        sel_cells = [c.lower() for c in args.cell_lines.split(',')] if args.cell_lines else cell_lines
        sel_drugs = [d.lower() for d in args.drugs.split(',')] if args.drugs else drugs
        pairs = list(itertools.product(sel_cells, sel_drugs))

    unknown_cells = sorted({c for c, _ in pairs} - set(cell_lines))
    unknown_drugs = sorted({d for _, d in pairs} - set(drugs))
    assert not unknown_cells, f"Cell lines not in the training dataset: {unknown_cells}"
    assert not unknown_drugs, f"Unknown drugs (add them with --new_drugs): {unknown_drugs}"
    return pairs


def predict(model, cell_idx, drug_idx, batch_size=4096):
    preds = []
    with torch.no_grad():
        for s in range(0, len(cell_idx), batch_size):
            i1 = torch.tensor(cell_idx[s:s + batch_size], dtype=torch.long, device=model.device)
            i2 = torch.tensor(drug_idx[s:s + batch_size], dtype=torch.long, device=model.device)
            preds.append(model.forward('cell_line-drug', i1, i2).squeeze(1).cpu().numpy())
            model.batches = {}
    return np.concatenate(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict drug responses with a trained NxtDRP model')
    parser.add_argument('--model', required=True, help='Model file saved by train.py')
    parser.add_argument('--pairs', default=None, help='csv with columns cell_line_name, drug_name')
    parser.add_argument('--cell_lines', default=None, help='Comma separated cell lines (default: all)')
    parser.add_argument('--drugs', default=None, help='Comma separated drugs (default: all)')
    parser.add_argument('--new_drugs', default=None, help='csv with columns drug_name, smiles of new compounds')
    parser.add_argument('--output', default='predictions.csv')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    ckpt, model = load_model(args.model, args.device)
    cell_lines, drugs = ckpt['cell_lines'], ckpt['drugs']
    n_known_drugs = len(drugs)
    if args.new_drugs:
        drugs = add_new_drugs(model, drugs, args.new_drugs)

    pairs = get_pairs(args, cell_lines, drugs)
    c_idx = {c: i for i, c in enumerate(cell_lines)}
    d_idx = {d: i for i, d in enumerate(drugs)}
    cell_idx = [c_idx[c] for c, _ in pairs]
    drug_idx = [d_idx[d] for _, d in pairs]

    y_hat = predict(model, cell_idx, drug_idx)

    observed = ckpt['observed'].tocsr()
    obs = np.array([observed[c, d] if d < n_known_drugs else 0 for c, d in zip(cell_idx, drug_idx)])

    out = pd.DataFrame({'cell_line_name': [c for c, _ in pairs],
                        'drug_name': [d for _, d in pairs],
                        'predicted': y_hat,
                        'observed': np.where(obs != 0, obs, np.nan)})
    if ckpt['target_transform'] == 'sigmoid':
        out['predicted_ln_ic50'] = score_to_ln_ic50(out['predicted'].values)
        out['observed_ln_ic50'] = score_to_ln_ic50(out['observed'].values)
    out.to_csv(args.output, index=False)
    print(f"{len(out)} predictions saved in {args.output}")
