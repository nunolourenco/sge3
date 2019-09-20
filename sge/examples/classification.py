import ast
import pandas as pd
from collections import Counter
from math import sin, cos, tan, sqrt, exp
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv, _sig_

import fairness.datasets
import fairness.utils


class BinaryClassification:
    def __init__(self, dataset_name, invalid_fitness=9999999):
        self.__train_set = []
        self.__test_set = []
        self.__invalid_fitness = invalid_fitness
        self.__dataset_name = dataset_name
        self.read_dataset(dataset_name)
        self.number_instances_per_class()

    def read_dataset(self, name):
        if name == 'adult-income':
            dataset = fairness.datasets.AdultIncomeDataset(encoding=fairness.utils.Encoding.ONE_HOT)
        elif name == 'german-credit':
            dataset = fairness.datasets.GermanCreditDataset(encoding=fairness.utils.Encoding.ONE_HOT)
        elif name == 'compas':
            dataset = fairness.datasets.CompasDataset(encoding=fairness.utils.Encoding.ONE_HOT)
        else:
            raise ValueError('unknown dataset')

        test_idx_df = pd.read_csv(dataset.DATA_PATH + '{}-{}-splits.csv'.format(dataset.name, dataset.encoding.value),
                                  header=0, index_col=0)

        test_idx = list(ast.literal_eval(test_idx_df.loc[1, 'TEST-IDX'].strip('[]')))

        self.__train_set = dataset.data.drop(index=test_idx).values
        self.__test_set = dataset.data.loc[test_idx, :].values

    def number_instances_per_class(self):
        self.__nc_train = Counter(self.__train_set[:, -1])
        self.__nc_test = Counter(self.__test_set[:, -1])

    def get_error(self, individual, dataset, nc):
        pred_error_per_class = dict()
        for c in nc.keys():
            pred_error_per_class[c] = 0.0

        for fit_case in dataset:
            case_output = fit_case[-1]
            try:
                result = eval(individual, globals(), {"x": fit_case[:-1]})
                pred_error_per_class[case_output] += (result - case_output)**2
            except (OverflowError, ValueError) as e:
                return self.__invalid_fitness

        pred_error = 1.0
        for c in nc.keys():
            pred_error *= exp(sqrt(pred_error_per_class[c] / nc[c]))

        return pred_error

    def evaluate(self, individual):
        if individual is None:
            return None

        train_error = self.get_error(individual, self.__train_set, self.__nc_train)
        test_error = self.get_error(individual, self.__test_set, self.__nc_test)

        return train_error, {'generation': 0, "evals": 1, "test_error": test_error}


if __name__ == "__main__":
    import sge
    eval_func = BinaryClassification(dataset_name='compas')
    sge.evolutionary_algorithm(evaluation_function=eval_func)
