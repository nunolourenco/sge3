import multiprocessing
import pandas as pd

from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC


def strategy_under(ratio):
    def ratio_multiplier(y):
        original_stats = Counter(y)
        min_samples = min(original_stats.values())
        target_stats = dict()
        for key, value in original_stats.items():
            if value == min_samples:
                target_stats[key] = min_samples
            else:
                target_stats[key] = int(ratio * min_samples)
        return target_stats
    return ratio_multiplier


def strategy_over(ratio):
    def ratio_multiplier(y):
        original_stats = Counter(y)
        max_samples = max(original_stats.values())
        target_stats = dict()
        for key, value in original_stats.items():
            if value == max_samples:
                target_stats[key] = max_samples
            else:
                target_stats[key] = int(ratio * max_samples)
        return target_stats
    return ratio_multiplier


def random_under_sampling(df, target, ratio=1.0, random_state=None):
    sampling_strategy = strategy_under(ratio)
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=False, random_state=random_state)
    # rus = RandomUnderSampler(sampling_strategy='auto', replacement=False, random_state=random_state)

    # target can be a Series or a str corresponding to one of the columns of a DataFrame
    if isinstance(target, pd.Series):
        rus.fit_resample(df, target)
    else:
        rus.fit_resample(df, df.loc[:, target])

    return rus.sample_indices_


def random_over_sampling(df, target, ratio=1.0, random_state=None):
    sampling_strategy = strategy_over(ratio)
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

    # target can be a Series or a str corresponding to one of the columns of a DataFrame
    if isinstance(target, pd.Series):
        ros.fit_resample(df, target)
    else:
        ros.fit_resample(df, df.loc[:, target])

    return ros.sample_indices_


def smote_categorical(df, cat_features, target, ratio, num_neighbors=5, random_state=None):
    num_jobs = multiprocessing.cpu_count() - 1

    sampling_strategy = strategy_over(ratio)
    smote_sampler = SMOTENC(categorical_features=cat_features,
                            sampling_strategy=sampling_strategy, k_neighbors=num_neighbors,
                            random_state=random_state, n_jobs=num_jobs)

    df_subset = df.iloc[:, cat_features].copy()

    # target can be a Series or a str corresponding to one of the columns of a DataFrame
    if isinstance(target, pd.Series):
        smote_sampler.fit_resample(df_subset, target)
    else:
        smote_sampler.fit_resample(df_subset, df.loc[:, target])

    return smote_sampler.sample_indices_


def sampling_method(df, target, ratio, random_state, type_sampler, cat_features=None, num_neighbors=5):
    if type_sampler == 'random-under':
        return random_under_sampling(df, target, ratio, random_state)
    elif type_sampler == 'random-over':
        return random_over_sampling(df, target, ratio, random_state)
    elif type_sampler == 'smote':
        if cat_features is None:
            raise ValueError('[ERROR] the indices of the categorical features must be specified')
        return smote_categorical(df, cat_features, target, ratio, num_neighbors, random_state)
    else:
        raise ValueError('[ERROR] unsupported sampling method')
