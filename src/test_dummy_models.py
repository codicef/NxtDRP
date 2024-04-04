#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
from sklearn import linear_model, metrics, model_selection, preprocessing, dummy

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import pickle



def dummy_model(STRATIFICATION="none"):


    df = pd.read_csv("./data/raw/relations/gdsc_drug_cellline_v6.csv")
    df['IC50'] = df['IC50'].apply(lambda x: 1/(1+np.exp(x)**(-0.1)))

    df['Max conc'] = df['Max conc'].apply(lambda x: 1/(1+x**(-0.1)))



    X = df[['Max conc']].values.reshape(-1, 1)

    X = df[['drug_name', 'cell_line_name', 'Max conc']]
    one_hot_drugs = pd.get_dummies(df['cell_line_name']).values
    one_hot_cell_lines = pd.get_dummies(df['drug_name']).values
    # X = np.concatenate([X, one_hot_drugs], axis=1)
    y = df['IC50'].values.reshape(-1, 1)
    drug_names = df['drug_name'].values
    cell_line_names = df['cell_line_name'].values


    if STRATIFICATION != "none":
        cv = model_selection.GroupShuffleSplit(n_splits=44, test_size=0.1, random_state=42)
    else:
        cv = model_selection.ShuffleSplit(n_splits=44, test_size=0.1, random_state=42)


    pearson = []
    r2 = []
    rmse = []

    pearson_drug_l = []
    r2_drug_l = []
    rmse_drug_l = []


    pearson_cell_l = []
    r2_cell_l = []
    rmse_cell_l = []

    groups = drug_names if STRATIFICATION == "drug_name" else cell_line_names
    if STRATIFICATION == "none":
        groups = None
    i = 0
    for train, test in cv.split(X, y, groups=groups):
        i += 1
        X_train, X_test = X.iloc[train], X.iloc[test]

        train_drugs = drug_names[train]
        test_drugs = drug_names[test]
        train_cell_lines = cell_line_names[train]
        test_cell_lines = cell_line_names[test]

        train_drugs_one_hot = one_hot_drugs[train]
        test_drugs_one_hot = one_hot_drugs[test]
        train_cell_lines_one_hot = one_hot_cell_lines[train]
        test_cell_lines_one_hot = one_hot_cell_lines[test]



        # print(f"Train drugs: {len(set(train_drugs))}")
        # print(f"Test drugs: {len(set(test_drugs))}")
        # print(f"Train cell lines: {len(set(train_cell_lines))}")
        # print(f"Test cell lines: {len(set(test_cell_lines))}")

        y_train, y_test = y[train], y[test].squeeze()

        try:
            if STRATIFICATION == "drug_name":
                df_filter = df[df['drug_name'].isin(train_drugs)]
                avg_ic50 = df_filter.groupby('cell_line_name')['IC50'].mean()
                X_avg = avg_ic50[test_cell_lines].values#.reshape(-1, 1)
                # X_avg = X_test['Max conc'].values.reshape(-1, 1)
            elif STRATIFICATION == "cell_line_name":
                df_filter = df[df['cell_line_name'].isin(train_cell_lines)]
                avg_ic50 = df_filter.groupby('drug_name')['IC50'].mean()
                X_avg = avg_ic50[test_drugs].values
            else:
                # use one hot for x
                X_train = np.concatenate([train_drugs_one_hot, train_cell_lines_one_hot], axis=1)
                X_test = np.concatenate([test_drugs_one_hot, test_cell_lines_one_hot], axis=1)

                model = linear_model.LinearRegression()
                model.fit(X_train, y_train)
                ypred = model.predict(X_test)
            # model = linear_model.Ridge(alpha=0.1)
            # model = linear_model.LinearRegression()
            # model = dummy.DummyRegressor(strategy="mean")

            # model.fit(X_train, y_train)

            if STRATIFICATION != "none":
                ypred = X_avg.squeeze() # model.predict(X_test)

            if metrics.root_mean_squared_error(y_test, ypred) > 100:
                continue
            pearson.append(pearsonr(ypred.squeeze(), y_test.squeeze())[0])
            r2.append(metrics.r2_score(y_test.squeeze(), ypred.squeeze()))
            rmse.append(metrics.root_mean_squared_error(y_test, ypred))

            # check perf by drug groups


            # Save preds ((train_cell, train_drug), train_true values, train_predicted values, (test_cell, test_drug), test_true values, test_predicted values)
            out = ((train_cell_lines, train_drugs), y_train, np.array([]), (test_cell_lines, test_drugs), y_test, ypred)

            with open(f"./log/preds/dummy_{STRATIFICATION}_{i}.pickle", 'wb') as f:
                pickle.dump(out, f)
                print(f"Saved preds to ./log/preds/dummy_{STRATIFICATION}_{i}.pickle")
        except Exception as e:
            print(e)
            i -= 1
            continue


if __name__ == '__main__':
    for STRATIFICATION in ["drug_name", "cell_line_name", "none"]:
        print(f"STRATIFICATION: {STRATIFICATION}")
        dummy_model(STRATIFICATION)
