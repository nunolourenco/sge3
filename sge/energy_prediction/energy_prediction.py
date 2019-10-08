import random
import pandas as pd
import numpy as np
from math import sin, cos, tan
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from scipy import optimize


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


class EnergyPrediction:
    def __init__(self, training_set_file=None, test_set_file=None, invalid_fitness=9999999):
        self.__train_set = []
        self.__test_set = None
        self.__number_of_variables = -1
        self.__invalid_fitness = invalid_fitness
        self.partition_rng = random.Random()
        self.test_set_file = test_set_file
        self.training_set_file = training_set_file
        self.read_fit_cases()


    def read_fit_cases(self):
        df = pd.read_csv(self.training_set_file, sep=";")
        self.__train_set = df.iloc[:, 1:].values  # ignore the year
        df = pd.read_csv(self.test_set_file, sep=";")
        self.__test_set = df.iloc[:, 1:].values  # ignore the year


    def get_error(self, individual, dataset):
        function = eval("lambda x, w: %s" % individual)

        def optimise_params(w):
            predicted = np.apply_along_axis(function, 0, dataset, w)
            pred_error = np.sum(np.power(predicted - dataset[:, 0], 2))
            return pred_error

        result = optimize.differential_evolution(optimise_params, bounds=[(-1, 1) for i in range(15)], maxiter=10)
        return result.fun, result.x

    def evaluate(self, individual):
        if individual is None:
            return None

        error, weights = self.get_error(individual, self.__train_set)
        error = _sqrt_(error / len(self.__train_set))

        if error is None:
            error = self.__invalid_fitness

        if self.test_set_file is not None:
            test_error, weights = self.get_error(individual, self.__test_set)
            test_error = _sqrt_(test_error / float(len(self.__test_set)))

        return error, {'generation': 0, "evals": 1, "test_error": test_error, "weigths": list(weights)}


if __name__ == "__main__":
    import sge
    eval_func = EnergyPrediction(training_set_file='energy_prediction/resources/Training_Data_Spain.csv',
                                 test_set_file='energy_prediction/resources/Test_Data_Spain.csv')
    sge.evolutionary_algorithm(evaluation_function=eval_func)
