"""
Metrics of (un)fairness.
"""
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score


def risk_difference(df, protected, target):
    num_protected = df[df[protected] == 0].shape[0]
    num_non_protected = df[df[protected] == 1].shape[0]

    num_protected_positive = df[(df[protected] == 0) & (df[target] == 1)].shape[0]
    num_non_protected_positive = df[(df[protected] == 1) & (df[target] == 1)].shape[0]

    return (num_non_protected_positive / num_non_protected) - (num_protected_positive / num_protected)


def disparate_impact(df, protected, target):
    num_protected = df[df[protected] == 0].shape[0]
    num_non_protected = df[df[protected] == 1].shape[0]

    num_protected_positive = df[(df[protected] == 0) & (df[target] == 1)].shape[0]
    num_non_protected_positive = df[(df[protected] == 1) & (df[target] == 1)].shape[0]

    return (num_protected_positive / num_protected) / (num_non_protected_positive / num_non_protected)


def generalized_entropy_index(df, true='label', prediction='prediction', alpha=2):
    df_copy = df.copy()

    # number of individuals
    n = df.shape[0]

    # calculate the benefit for each individual
    df_copy['benefit'] = df_copy.apply(lambda row: 1 + row[prediction] - row[true], axis=1)

    # calculate mean benefit
    mean_benefit = df_copy['benefit'].mean()

    temp = df_copy.apply(lambda row: (row['benefit'] / mean_benefit)**alpha - 1, axis=1)

    # calculate the generalized entropy index
    ge = temp.sum() / (n * alpha * (alpha - 1))

    return ge


def compute_all_types_agg(metric_by_sensitive_attribute):
    if metric_by_sensitive_attribute[1] != 0:
        metric_ratio = metric_by_sensitive_attribute[0] / metric_by_sensitive_attribute[1]
    else:
        metric_ratio = np.NaN

    results_dict = {'mean': np.mean(metric_by_sensitive_attribute.values()),
                    'diff': metric_by_sensitive_attribute[1] - metric_by_sensitive_attribute[0],
                    'ratio': metric_ratio}

    return results_dict


def tpr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if tp + fn != 0:
        return tp / (tp + fn)
    else:
        return np.NaN


def tnr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if tn + fp != 0:
        return tn / (tn + fp)
    else:
        return np.NaN


def bcr_score(y_true, y_pred):
    tpr = tpr_score(y_true, y_pred)
    tnr = tnr_score(y_true, y_pred)

    if tpr == np.NaN or tnr == np.NaN:
        return np.NaN
    else:
        return (tpr + tnr) / 2


def f1_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return (2 * tp) / (2 * tp + fn + fp)


def conditioned_metric(metric_function, df, protected, true='label', prediction='prediction', type_agg='mean'):
    metric = dict()

    sensitive_values = df[protected].unique().tolist()

    for val in sensitive_values:
        df_subset = df[df[protected] == val]
        metric[val] = metric_function(df_subset[true], df_subset[prediction])

    if type_agg == 'mean':
        return np.mean(list(metric.values()))
    elif type_agg == 'diff':
        return metric[1] - metric[0]
    elif type_agg == 'ratio':
        if metric[1] != 0:
            return metric[0] / metric[1]
        else:
            return np.NaN
    elif type_agg == 'all':
        return compute_all_types_agg(metric)
    else:
        raise ValueError('[ERROR] unknown type of metric')


def conditioned_acc(df, protected, true='label', prediction='prediction', type_agg='mean'):
    return conditioned_metric(accuracy_score, df, protected, true, prediction, type_agg)


def conditioned_tpr(df, protected, true='label', prediction='prediction', type_agg='mean'):
    return conditioned_metric(tpr_score, df, protected, true, prediction, type_agg)


def conditioned_tnr(df, protected, true='label', prediction='prediction', type_agg='mean'):
    return conditioned_metric(tnr_score, df, protected, true, prediction, type_agg)


def conditioned_bcr(df, protected, true='label', prediction='prediction', type_agg='mean'):
    return conditioned_metric(bcr_score, df, protected, true, prediction, type_agg)


def conditioned_f1(df, protected, true='label', prediction='prediction', type_agg='mean'):
    return conditioned_metric(f1_score, df, protected, true, prediction, type_agg)
