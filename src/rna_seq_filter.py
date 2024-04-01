#!/usr/bin/env python3

import pandas as pd
import seaborn as sns


df = pd.read_csv("./data/raw/relations/rnaseq_tpm_cellline_v6.csv")

df = df.dropna(axis=1, how='all')

df = df.dropna(axis=0, how='all')

# Keep top k genes by variance on 'tpm' values, and then keep the same structure at the end
k = 500
df2 = df.groupby('gene_symbol').var().sort_values(by='tpm', ascending=False).head(k)
df2 = df2.reset_index()
df = df[df['gene_symbol'].isin(df2['gene_symbol'])]


print(df.head())
print(f"Number of genes: {len(df['gene_symbol'].unique())}")
print(f"Number of cell lines: {len(df['cell_line_name'].unique())}")
print(f"Number of samples: {len(df)}")
df.to_csv("./data/raw/relations/rnaseq_tpm_cellline_v6_top500.csv", index=False)
