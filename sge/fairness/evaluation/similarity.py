"""
Metrics and functions to evaluate the similarity between features.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import entropy


def mutual_information_discrete(first_variable, second_variable):
    # contingency_table = pd.crosstab(first_variable, second_variable)
    contingency_table = contingency_matrix(first_variable, second_variable, sparse=True)
    mi = mutual_info_score(None, None, contingency=contingency_table)
    return mi


def normalized_mutual_information_discrete(first_variable, second_variable):
    """ Adapted from the implementation provided by scikit-learn. """
    mi = mutual_information_discrete(first_variable, second_variable)
    h_first, h_second = entropy(first_variable), entropy(second_variable)
    normalizer = np.sqrt(h_first * h_second)
    normalizer = max(normalizer, np.finfo('float64').eps)
    nmi = mi / normalizer
    return nmi


def cramers_v(first_variable, second_variable):
    """
    Computes Cramer's V with bias correction.

    Adapted from: https://github.com/shakedzy/dython/blob/master/dython/nominal.py
    """
    contingency_table = contingency_matrix(first_variable, second_variable, sparse=True).toarray()
    r, k = contingency_table.shape

    chi2 = ss.chi2_contingency(contingency_table, correction=False)[0]
    n = contingency_table.sum()
    phi2 = chi2 / n

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def pairwise_cramers_v(df, target=None, visualization=False, out_filename=None):
    if target is None:
        nfeatures = df.shape[1]

        v = np.zeros((nfeatures, nfeatures))

        for i in range(nfeatures):
            for j in range(nfeatures):
                v[i, j] = cramers_v(df.iloc[:, i], df.iloc[:, j])

        if visualization:
            sns.set_context('talk')

            mask = np.zeros_like(v, dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            mask[np.diag_indices_from(mask)] = False

            # INTEGER: 15; COMPAS ONE_HOT: 20; GERMAN ONE_HOT: 25
            plt.figure(figsize=(35, 35))

            # draw the heatmap with the mask and correct aspect ratio
            ax = sns.heatmap(v, vmin=0.0, center=0.5, vmax=1.0, mask=mask, cmap='Reds',
                             xticklabels=df.columns, yticklabels=df.columns,
                             square=True, linewidths=.5, cbar_kws={"shrink": .5})

            ax.invert_yaxis()

            plt.yticks(rotation=0)

            ax.get_yticklabels()[-2].set_fontweight('bold')
            ax.get_yticklabels()[-2].set_color('red')
            ax.get_xticklabels()[-2].set_fontweight('bold')
            ax.get_xticklabels()[-2].set_color('red')

            plt.savefig(out_filename, bbox_inches='tight')
    else:
        v_dict = dict()

        for feature, series in df.iteritems():
            v_dict[feature] = cramers_v(series, target)

        v = pd.Series(data=v_dict)

    return v


def pairwise_mutual_information(df, target=None, visualization=False, out_filename=None):
    if target is None:
        nfeatures = df.shape[1]

        mi = np.zeros((nfeatures, nfeatures))

        for i in range(nfeatures):
            for j in range(nfeatures):
                mi[i, j] = normalized_mutual_information_discrete(df.iloc[:, i], df.iloc[:, j])

        if visualization:
            sns.set_context('talk')

            mask = np.zeros_like(mi, dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            mask[np.diag_indices_from(mask)] = False

            # get the maximum entropy
            # max_mi = mi[~mask].max()

            # INTEGER: 15; COMPAS ONE_HOT: 20; GERMAN ONE_HOT: 25
            plt.figure(figsize=(35, 35))

            # draw the heatmap with the mask and correct aspect ratio
            ax = sns.heatmap(mi, vmin=0.0, center=0.5, vmax=1.0, mask=mask, cmap='Reds',
                             xticklabels=df.columns, yticklabels=df.columns,
                             square=True, linewidths=.5, cbar_kws={"shrink": .5})

            ax.invert_yaxis()

            plt.yticks(rotation=0)

            ax.get_yticklabels()[-2].set_fontweight('bold')
            ax.get_yticklabels()[-2].set_color('red')
            ax.get_xticklabels()[-2].set_fontweight('bold')
            ax.get_xticklabels()[-2].set_color('red')

            plt.savefig(out_filename, bbox_inches='tight')
    else:
        # nfeatures = df.shape[1]

        # mi = [normalized_mutual_information_discrete(df.iloc[:, feature], target) for feature in range(nfeatures)]

        mi_dict = dict()

        for feature, series in df.iteritems():
            mi_dict[feature] = normalized_mutual_information_discrete(series, target)

        mi = pd.Series(data=mi_dict)

    return mi


def pairwise_correlation(df, target=None, method='pearson', visualization=False, out_filename=None):
    if target is None:
        corr = df.corr(method=method)

        if visualization:
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            mask[np.diag_indices_from(mask)] = False

            plt.figure(figsize=(20, 20))

            # generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # draw the heatmap with the mask and correct aspect ratio
            ax = sns.heatmap(corr, vmin=-1.0, vmax=1.0, center=0.0,
                             xticklabels=df.columns, yticklabels=df.columns,
                             mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})

            ax.invert_yaxis()

            plt.yticks(rotation=0)

            plt.savefig(out_filename, bbox_inches='tight')
    else:
        if isinstance(target, pd.Series):
            target_series = target.copy()
        else:
            target_series = pd.Series(data=target, index=df.index)

        # nfeatures = df.shape[1]
        # corr = [target_series.corr(df.iloc[:, feature], method=method) for feature in range(nfeatures)]

        corr_dict = dict()

        for feature, series in df.iteritems():
            corr_dict[feature] = target_series.corr(series, method=method)

        corr = pd.Series(data=corr_dict)

    return corr


def save_similarity_series(filename, series, run, fold):
    series.loc['RUN'] = run
    series.loc['FOLD'] = fold
    df = pd.DataFrame(series).T.set_index(['RUN', 'FOLD'])
    if run == 1 and fold == 1:
        df.to_csv(path_or_buf=filename)
    else:
        df.to_csv(path_or_buf=filename, header=False, mode='a')
