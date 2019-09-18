import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import normalized_mutual_info_score as nmi

from fairness.evaluation import risk_difference, disparate_impact
from fairness.utils import Encoding, convert_to_one_hot_encoding


class Dataset:
    DATA_PATH = 'data/'

    def __init__(self, name=None, data=None, columns=None, protected=None, target=None, classes=None, encoding=None):
        self.name = name
        self.data = data
        self.columns = columns
        self.features = None
        self.features_without_protected = None
        self.protected = protected
        self.target = target
        self.classes = classes
        self.encoding = encoding

    def remove_columns(self, columns_to_remove):
        if self.columns:
            self.data.drop(columns=columns_to_remove, inplace=True)
            self.columns = self.data.columns.tolist()
            self.features_including_protected()
            self.features_excluding_protected()
        else:
            raise ValueError('[ERROR] the dataset has no columns defined yet')

    def convert_to_default_format(self):
        """ Reformat a DataFrame such that the true labels correspond to the last column
        and the protected attributes are the columns immediately before the last column.
        """

        self.columns.remove(self.protected)
        self.columns.remove(self.target)

        # append protected and target attributes to the last columns
        self.columns.append(self.protected)
        self.columns.append(self.target)

        # reformat the DataFrame
        self.data = self.data.reindex(labels=self.columns, axis=1, copy=False)

    def class_names_sorted_by_value(self):
        return [value for key, value in sorted(self.classes.items())]

    def features_including_protected(self):
        if self.columns:
            self.features = np.delete(self.columns, self.columns.index(self.target))
        else:
            raise ValueError('[ERROR] the dataset has no columns defined yet')

    def features_excluding_protected(self):
        if self.columns:
            self.features_without_protected = np.delete(self.columns, [self.columns.index(self.target),
                                                                       self.columns.index(self.protected)])
        else:
            raise ValueError('[ERROR] the dataset has no columns defined yet')

    def get_max_num_categories(self, include_protected):
        if include_protected:
            return max(self.data.loc[:, self.features].nunique().tolist())
        else:
            return max(self.data.loc[:, self.features_without_protected].nunique().tolist())

    def compute_basic_stats(self, dropna=False):
        print('\n>>>> {} with {} encoding - basic statistics'.format(self.name, self.encoding.name))

        nsamples = self.data.shape[0]

        if dropna:
            data = self.data.dropna(inplace=False)
            nsamples_dropped = nsamples - data.shape[0]
            if nsamples_dropped > 0:
                print('[WARNING] {} samples contained missing values and were dropped'.format(nsamples_dropped))
        else:
            data = self.data.copy()

        print(data[self.protected].value_counts())

        print(data[self.target].value_counts())

        print(data.groupby(self.protected)[self.target].value_counts())

        print(data.groupby(self.target)[self.protected].value_counts())

    def compute_fairness_metrics(self, dropna=False):
        print('\n>>>> {} with {} encoding - fairness metrics'.format(self.name, self.encoding.name))

        nsamples = self.data.shape[0]

        if dropna:
            data = self.data.dropna(inplace=False)
            nsamples_dropped = nsamples - data.shape[0]
            if nsamples_dropped > 0:
                print('[WARNING] {} samples contained missing values and were dropped'.format(nsamples_dropped))
        else:
            data = self.data.copy()

        print('[{} - {} encoding] CVS: {}'.format(self.name, self.encoding.name,
                                                  risk_difference(data, protected=self.protected, target=self.target)))

        print('[{} - {} encoding] NPI (true labels & protected attribute): {}'.format(self.name, self.encoding.name,
                                                                                      nmi(data.loc[:, self.target],
                                                                                          data.loc[:, self.protected],
                                                                                          average_method='geometric')))

        print('[{} - {} encoding] DI: {}'.format(self.name, self.encoding.name,
                                                 disparate_impact(data, protected=self.protected, target=self.target)))

    def combine_two_attributes(self, first: str, second: str):
        first_values = sorted(self.data[first].unique())
        second_values = sorted(self.data[second].unique())

        first_str = list(map(str, first_values))
        second_str = list(map(str, second_values))

        categories = list(map(lambda x: x[0] + x[1], itertools.product(first_str, second_str)))

        series = self.data.apply(lambda row: str(int(row[first])) + str(int(row[second])), axis=1)
        series = series.astype('category').rename(first + '_' + second)

        new_categories = dict()
        for idx, category in enumerate(categories):
            new_categories[category] = idx

        series.cat.rename_categories(new_categories, inplace=True)

        if self.encoding is Encoding.INTEGER:
            df = pd.DataFrame(series, index=series.index)
        elif self.encoding is Encoding.ONE_HOT:
            df = convert_to_one_hot_encoding(series, categories=categories)
        else:
            raise ValueError('[ERROR] unsupported encoding')

        return df, categories

    def partition_dataset(self, splits, test_size=0.3, seed=None):
        data = pd.DataFrame()

        splitter = ShuffleSplit(n_splits=splits, test_size=test_size, random_state=seed)

        run_number = 0

        for train_idx, test_idx in splitter.split(self.data):
            run_number += 1

            new_row = {
                'RUN': run_number,
                'TEST-IDX': self.data.iloc[test_idx].index.tolist()
            }

            data = data.append(new_row, ignore_index=True)

        data['RUN'] = data['RUN'].astype('int64')
        data.set_index('RUN', inplace=True)

        data.to_csv(self.DATA_PATH + '{}-{}-splits.csv'.format(self.name, self.encoding.value))
