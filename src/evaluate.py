#!/usr/bin/env python3

from sklearn import metrics
from scipy import stats
import seaborn as sns
from matplotlib import pyplot

precision = 4

def evaluate_regression(y_hat, y_true, plot=False):
    perf_d = {}

    perf_d['mse'] = metrics.mean_squared_error(y_true, y_hat, squared=True)
    perf_d['rmse'] = metrics.mean_squared_error(y_true, y_hat, squared=False)
    perf_d['r2'] = metrics.r2_score(y_true, y_hat)
    perf_d['spearman'] =  stats.spearmanr(y_true, y_hat)[0]
    perf_d['pearson'] = stats.pearsonr(y_true, y_hat)[0]

    for key in perf_d:
        perf_d[key] = round(perf_d[key], precision)

    return perf_d
