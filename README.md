# NxtDRP

NxtDRP predicts the drug response of cancer cell lines by integrating multi-omics data (RNA-Seq and Proteomics) and the molecular structure of the drugs. It is built on the [NXTfusion](https://doi.org/10.1093/bioinformatics/btab092) non-linear data fusion library: cell lines, drugs, genes and proteins are entities of an Entity-Relation graph, each omics is a relation, and a Graph Neural Network encodes the molecular graph of each drug.

This repository lets you:
- **reproduce** the experiments of the paper on GDSC (and run the same protocol on CCLE);
- **use** a trained model to predict the response of cell lines to known drugs or to new compounds given as SMILES;
- **evaluate** any drug response predictor with the validation protocol of the paper (Global, Fixed-Drug and Fixed-Cell Line aggregation).

> Codicè F, Pancotti C, Rollo C, Moreau Y, Fariselli P, Raimondi D. *The specification game: rethinking the evaluation of drug response prediction for precision oncology.* Journal of Cheminformatics (2025). [doi:10.1186/s13321-025-00972-y](https://doi.org/10.1186/s13321-025-00972-y)

## Contents
1. [Installation](#1-installation)
2. [Data](#2-data)
3. [Quickstart](#3-quickstart)
4. [Reproducing the paper](#4-reproducing-the-paper)
5. [Using the model](#5-using-the-model)
6. [Evaluating any DRP predictor](#6-evaluating-any-drp-predictor)
7. [Notes](#7-notes)

All commands must be run from the repository root.

## 1. Installation
```bash
git clone https://github.com/codicef/NxtDRP.git
cd NxtDRP
conda env create -f environment.yml
conda activate nxtdrp
```
All package versions are pinned (Python 3.10, PyTorch 1.13 with CUDA 11.7, PyTorch Geometric 2.6.1). A GPU is recommended but everything also runs on CPU (`--device cpu`).

## 2. Data
Download the preprocessed data from the [dataset release](https://github.com/codicef/NxtDRP/releases/tag/dataset) and build the datasets:
```bash
bash download_data.sh
python src/data.py --dataset all
```

| Dataset | Target | Cell lines | Drugs | Response values | Omics |
|---|---|---|---|---|---|
| `gdsc` | IC50 | 948 | 223 | 172,114 | RNA-Seq (500 genes), Proteomics (8,457 proteins) |
| `gdsc_auc` | AUDRC | 948 | 223 | 172,114 | as `gdsc` |
| `ccle` | IC50 | 504 | 24 | 11,670 | RNA-Seq (500 genes), Proteomics (12,755 proteins) |

Preprocessing (as in the paper):
- GDSC v6 IC50 values with a curve fit RMSE > 0.3 are discarded, only drugs with a PubChem structure are kept;
- IC50 values (provided as ln IC50) are rescaled to [0, 1] with `y = 1 / (1 + IC50^-0.1)`; AUDRC is already in [0, 1];
- RNA-Seq (log TPM, most variable genes) and Proteomics values are min-max scaled to [0, 1];
- each drug is a molecular graph: atoms are nodes (symbol, atomic number, degree, formal charge, ring, radical electrons, hydrogens, aromaticity), bonds are edges.

<details>
<summary>Raw files and how they are generated</summary>

`download_data.sh` creates:

| File | Content |
|---|---|
| `data/raw/relations/gdsc_drug_cellline_v6.csv` | GDSC drug response (ln IC50, AUDRC, Max conc) |
| `data/raw/relations/rnaseq_tpm_cellline_v6_top1000.csv` | GDSC RNA-Seq TPM, most variable genes |
| `data/raw/relations/protein_zscore_cellline_v6_l.csv` | GDSC Proteomics |
| `data/raw/entities/drugs_v6.csv` | GDSC drug molecular graphs |
| `data/raw/relations/ccle_drug_response.csv` | CCLE drug response (ln IC50, Max conc) |
| `data/raw/relations/ccle_rnaseq_top500.csv` | CCLE RNA-Seq TPM, 500 most variable genes |
| `data/raw/relations/ccle_proteomics.csv` | CCLE Proteomics |
| `data/raw/entities/drugs_ccle.csv` | CCLE drug molecular graphs |

The CCLE files are produced from the original CCLE tables with `python src/preprocess_ccle.py` (see `--help` for the input paths). Drug molecular graphs can be computed from any csv with `drug_name` and `smiles` columns with `python src/drugs_encoding.py --input drugs.csv --output drugs_encoded.csv`. The most variable genes of an RNA-Seq table are selected with `src/rna_seq_filter.py`.
</details>

## 3. Quickstart
A short run on CCLE (2 train/test splits, unseen cell lines):
```bash
python src/main.py --dataset ccle --cv_type unseen_cell --n_tests 2
```
The predictions of each split are saved in `results/ccle_NxtDRP_pr_ex_unseen_cell/` and the metrics of the three aggregation strategies are printed at the end of the log in `log/`.

## 4. Reproducing the paper
Each experiment trains and tests NxtDRP on 40 random 90/10 train/test splits, built with one of three splitting strategies, and evaluates the predictions with three aggregation strategies.

```bash
python src/main.py --dataset gdsc --model NxtDRP --omics pr_ex --cv_type unseen_cell --n_tests 40
```

| Option | Values |
|---|---|
| `--dataset` | `gdsc`, `gdsc_auc`, `ccle` |
| `--model` | `NxtDRP`, or `NxtDRPMC` (adds the maximum tested concentration of the drug, MT+MC in the paper) |
| `--omics` | ER graph: `none` (MT), `pr` (MT+PR), `ex` (MT+EX), `pr_ex` (MT+PR+EX, default) |
| `--cv_type` | `random_split`, `unseen_cell`, `unseen_drug` |
| `--n_tests` | number of train/test splits (40 in the paper) |
| `--default_hp_path` | hyperparameters of each splitting strategy (default `data/hyperparameters/default_hp.json`, tuned on GDSC) |
| `--optimize_hp` | tune the hyperparameters with Optuna on each split instead (much slower) |
| `--seed` | seed of the splits and of the model initialization (default `1956`) |
| `--device` | `cuda` or `cpu` |

Outputs of an experiment, in `results/<dataset>_<model>_<omics>_<cv_type>/`:
- `split_XX.csv`: test predictions of each split, with columns `cell, drug, true_value, predicted_value`;
- `metrics.json`: RMSE, R2, Spearman and Pearson of each split for the `global`, `fixed_drug` and `fixed_cell` aggregations.

The dummy baselines (DummyDrugAvg, DummyCellAvg, DummyLR, DummyMC) use the same splitting strategies and output format:
```bash
python src/dummy_models.py --dataset gdsc
```

To run all the GDSC experiments of the paper and collect them in `results/summary.csv`:
```bash
bash scripts/reproduce_paper.sh               # 40 splits per experiment
N_TESTS=2 bash scripts/reproduce_paper.sh     # quick check
```
Running time: one split on GDSC takes about 15-20 minutes on an RTX 3090 (190 epochs), so a 40-split experiment takes about 12 hours and the full script several days. The dummy baselines take a few minutes.

The predictions used for the figures of the paper (NxtDRP, tCNN, GraphDRP and dummy models) are available in [codicef/DRPValidation](https://github.com/codicef/DRPValidation/tree/main/predictions) and can be evaluated with `src/validation.py` (Section 6).

## 5. Using the model
### Train a final model
Train NxtDRP on all the drug response values of a dataset:
```bash
python src/train.py --dataset gdsc --omics pr_ex
```
The model is saved in `models/nxtdrp_gdsc_pr_ex.pt`. Use `--hp_key` to pick the hyperparameter set (`cell` random split, default; `row` unseen cell lines; `col` unseen drugs).

### Predict
```bash
# all cell lines x all drugs of the dataset
python src/predict.py --model models/nxtdrp_gdsc_pr_ex.pt --output predictions.csv

# selected cell lines and drugs
python src/predict.py --model models/nxtdrp_gdsc_pr_ex.pt --cell_lines A549,MCF7 --drugs Erlotinib,Lapatinib

# specific pairs (csv with columns cell_line_name, drug_name)
python src/predict.py --model models/nxtdrp_gdsc_pr_ex.pt --pairs pairs.csv

# new compounds (csv with columns drug_name, smiles)
python src/predict.py --model models/nxtdrp_gdsc_pr_ex.pt --new_drugs new_drugs.csv --drugs my_compound
```
The output csv contains, for each pair, `predicted` (the [0, 1] score), `predicted_ln_ic50` (converted back to ln IC50 for IC50 models), and the measured values (`observed`, `observed_ln_ic50`) when the pair is in the dataset. Names are matched case-insensitively and written in lowercase.

**New cell lines**: NxtDRP learns a representation for each cell line from its omics and drug responses, so cell lines must be part of the training dataset. To predict a new cell line, add its RNA-Seq and/or Proteomics values to the raw files, rebuild the dataset with `src/data.py` and retrain: the cell line will be represented through its omics even without any drug response.

**What to expect**: as discussed in the paper, predictions for known drugs on cell lines with omics are informative (Fixed-Drug Pearson r of about 0.33 on unseen GDSC cell lines), while predictions for new compounds are much less reliable (Fixed-Cell Line r of about 0.28 on unseen drugs). Use the validation protocol below to check the performance in the setting you care about.

## 6. Evaluating any DRP predictor
`src/validation.py` (from [codicef/DRPValidation](https://github.com/codicef/DRPValidation)) evaluates predictions of any method. Save one csv per train/test split with the columns `cell, drug, true_value, predicted_value` and run:
```bash
python src/validation.py evaluate my_method_predictions/ --save_metrics
python src/validation.py evaluate results/*/ --summary results/summary.csv   # compare several runs
```
Metrics are computed with three aggregation strategies:
- **Global**: over the whole test set (the usual approach, inflated by the differences among drugs);
- **Fixed-Drug**: for each drug, then averaged; measures the ability to rank cell lines and is the relevant one for **unseen cell lines**;
- **Fixed-Cell Line**: for each cell line, then averaged; measures the ability to rank drugs and is the relevant one for **unseen drugs**.

Predictions in the legacy pickle format can be converted with `python src/validation.py convert <pickle_dir> <csv_dir>`.

## 7. Notes
- **Reproducibility**: with the same `--seed`, runs on CPU are bit-for-bit identical. On GPU some PyTorch Geometric operations are not deterministic, so results vary slightly between runs.
- Correlations are undefined (`NaN`) for a group whose predictions are constant, e.g. DummyDrugAvg in the Fixed-Drug aggregation; this corresponds to no ranking ability (r = 0 in the paper).
- When `--optimize_hp` is used, the hyperparameter validation folds are drawn from the whole drug response matrix, including the test pairs of the current split.

### Repository structure
```
src/
  data.py               build the serialized ER-graph datasets
  main.py               randomized train/test evaluation of NxtDRP
  models.py             NxtDRP and NxtDRPMC models
  dummy_models.py       dummy baselines
  validation.py         Global / Fixed-Drug / Fixed-Cell Line validation
  train.py, predict.py  final model training and prediction
  drugs_encoding.py     SMILES -> molecular graphs
  preprocess_ccle.py, rna_seq_filter.py   raw data preprocessing
  NXTfusion/            NXTfusion library
scripts/reproduce_paper.sh
data/hyperparameters/default_hp.json
```

## License
This project is licensed under the MIT License.
