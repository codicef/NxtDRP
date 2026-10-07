#!/usr/bin/env python3
'''
Converts the original CCLE tables into the raw input files read by data.py
(same layout and preprocessing as the GDSC files of the dataset release).
'''

import argparse
from os import path, makedirs
import pandas as pd
from rna_seq_filter import filter_top_variable_genes
from drugs_encoding import encode_drugs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare CCLE raw files for data.py')
    parser.add_argument('--ccle_dir', default='./data/raw/relations/ccle',
                        help='Folder with ccle_drug_response.csv, ccle_rnaseq.csv, ccle_proteomics.csv')
    parser.add_argument('--drugs_smiles', default='./data/raw/entities_info/drugs_ccle.csv',
                        help='csv with drug_name and smiles of the CCLE drugs')
    parser.add_argument('--raw_dir', default='./data/raw')
    parser.add_argument('--n_genes', type=int, default=500)
    args = parser.parse_args()

    rel_dir = path.join(args.raw_dir, 'relations')
    ent_dir = path.join(args.raw_dir, 'entities')
    makedirs(rel_dir, exist_ok=True)
    makedirs(ent_dir, exist_ok=True)

    # Drug response (IC50 already in natural log scale, as in GDSC)
    dr = pd.read_csv(path.join(args.ccle_dir, 'ccle_drug_response.csv'))
    dr = dr[['drug_name', 'cell_line_name', 'IC50', 'Max conc']].dropna()
    dr.to_csv(path.join(rel_dir, 'ccle_drug_response.csv'), index=False)
    print(f"Drug response: {len(dr)} pairs, {dr['drug_name'].nunique()} drugs, "
          f"{dr['cell_line_name'].nunique()} cell lines")

    # Drug molecular graphs
    encode_drugs(args.drugs_smiles, path.join(ent_dir, 'drugs_ccle.csv'), smiles_key='smiles')

    # Proteomics
    pr = pd.read_csv(path.join(args.ccle_dir, 'ccle_proteomics.csv'))
    pr = pr[['uniprot_id', 'z-score', 'cell_line_name']].dropna()
    pr = pr[pr['cell_line_name'].isin(dr['cell_line_name'])]
    pr.to_csv(path.join(rel_dir, 'ccle_proteomics.csv'), index=False)
    print(f"Proteomics: {len(pr)} values")

    # RNA-Seq, most variable genes
    filter_top_variable_genes(path.join(args.ccle_dir, 'ccle_rnaseq.csv'),
                              path.join(rel_dir, f'ccle_rnaseq_top{args.n_genes}.csv'),
                              k=args.n_genes, cell_lines=dr['cell_line_name'])
