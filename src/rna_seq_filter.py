#!/usr/bin/env python3

import argparse
import pandas as pd


def filter_top_variable_genes(in_file, out_file, k=500, cell_lines=None):
    '''
    Keep the top k genes by variance of their 'tpm' values across cell lines
    '''
    df = pd.read_csv(in_file, usecols=['cell_line_name', 'gene_symbol', 'tpm'])
    df = df.dropna()
    if cell_lines is not None:
        df = df[df['cell_line_name'].isin(cell_lines)]

    top_genes = df.groupby('gene_symbol')['tpm'].var().sort_values(ascending=False).head(k)
    df = df[df['gene_symbol'].isin(top_genes.index)]

    print(f"Number of genes: {len(df['gene_symbol'].unique())}")
    print(f"Number of cell lines: {len(df['cell_line_name'].unique())}")
    print(f"Number of samples: {len(df)}")
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select the most variable genes of an RNA-Seq TPM table')
    parser.add_argument('--input', default='./data/raw/relations/rnaseq_tpm_cellline_v6.csv')
    parser.add_argument('--output', default='./data/raw/relations/rnaseq_tpm_cellline_v6_top1000.csv')
    parser.add_argument('--k', type=int, default=500)
    args = parser.parse_args()

    filter_top_variable_genes(args.input, args.output, args.k)
