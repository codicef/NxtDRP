#!/usr/bin/env python3
'''
Train a final NxtDRP model on all the drug response values of a dataset and save it,
to be used by predict.py.
'''

import argparse
import json
import os
import torch
from data import DatasetHandler, load_dataset_info
from models import NxtDRP
from main import OMICS_RELATIONS, get_losses_dict, set_seed
from NXTfusion.NXmultiRelSide import NNwrapper


def train_final_model(dataset_path, omics, hyperparameters, device, seed):
    set_seed(seed)
    ds = DatasetHandler.load_serialized(dataset_path, load_side_info=False,
                                        relations_to_load=OMICS_RELATIONS[omics])
    er, _, _, _ = ds.get_er_graph(get_losses_dict())

    model = NxtDRP(er, 'final', main_emb_size=hyperparameters['emb_size'],
                   dropout=hyperparameters['dropout'],
                   gnn_size=hyperparameters['gnn_size'],
                   out_emb_size=hyperparameters['out_emb_size'],
                   device=device,
                   side_info_file=os.path.join(dataset_path, 'entities', 'drug.csv'))

    wrapper = NNwrapper(model, dev=device, ignore_index=-1)
    wrapper.fit(er, LOG=False, epochs=hyperparameters['epochs'],
                weight_decay=hyperparameters['weight_decay'],
                batch_size=hyperparameters['batch_size'])
    return model, ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a final NxtDRP model on a whole dataset')
    parser.add_argument('--dataset', type=str, default='gdsc',
                        help='Dataset built by src/data.py: gdsc, gdsc_auc, ccle or a custom one')
    parser.add_argument('--datasets_dir', type=str, default='data/datasets')
    parser.add_argument('--omics', type=str, default='pr_ex', choices=list(OMICS_RELATIONS.keys()))
    parser.add_argument('--hp_path', type=str, default='data/hyperparameters/default_hp.json')
    parser.add_argument('--hp_key', type=str, default='cell',
                        help='Hyperparameter set in --hp_path: cell (random split), row (unseen cell), col (unseen drug)')
    parser.add_argument('--epochs', type=int, default=None, help='Override the number of epochs')
    parser.add_argument('--output', type=str, default=None,
                        help='Model file (default models/nxtdrp_<dataset>_<omics>.pt)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1956)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = 'cpu'

    with open(args.hp_path) as f:
        hyperparameters = json.load(f)[args.hp_key]
    if args.epochs is not None:
        hyperparameters['epochs'] = args.epochs

    dataset_path = os.path.join(args.datasets_dir, args.dataset)
    model, ds = train_final_model(dataset_path, args.omics, hyperparameters, args.device, args.seed)

    output = args.output or os.path.join('models', f"nxtdrp_{args.dataset}_{args.omics}.pt")
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    model.batches = {}
    info = load_dataset_info(dataset_path)
    torch.save({'model': model,
                'dataset': args.dataset,
                'target': info['target'],
                'target_transform': info['transform'],
                'omics': args.omics,
                'hyperparameters': hyperparameters,
                'cell_lines': list(ds.entities_dict['cell_line'].idx_e),
                'drugs': list(ds.entities_dict['drug'].idx_e),
                'observed': ds.rel_dict['cell_line-drug'].matrix},
               output)
    print(f"Model saved in {output}")
