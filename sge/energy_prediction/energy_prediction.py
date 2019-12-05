import random
import pandas as pd
import numpy as np
from math import sin, cos, tan
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from scipy import optimize
import yabox


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return np.abs(_error(actual, predicted)) / (actual + EPSILON)

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
        actual = dataset[:, 0]

        def optimise_params(w):
            predicted = np.apply_along_axis(function, 1, dataset, w)
            # pred_error = np.mean(np.sqrt(np.power(_error(actual, predicted), 2)))
            pred_error = np.sum(np.abs(_error(actual, predicted)))
            return pred_error

        # result = optimize.differential_evolution(optimise_params, bounds=[(0, 1) for i in range(15)], maxiter=100,
        #                                        disp=True, popsize=75, mutation=0.4717, recombination=0.8803)
        result = optimize.differential_evolution(optimise_params, bounds=[(-1, 1) for i in range(15)], maxiter=100,
                                                 popsize=75, mutation=0.4717, recombination=0.8803, disp=False)
        # de = yabox.DE(optimise_params, [(0, 1) for i in range(15)], mutation=0.4717, crossover=0.8803, maxiters=100,
         #              popsize=100)
        # w, f = de.solve(show_progress=False)

        return result.fun, result.x

    def get_test_error(self, individual, dataset, weights):
        function = eval("lambda x, w: %s" % individual)
        actual = dataset[:, 0]
        predicted = np.apply_along_axis(function, 1, dataset, weights)
        pred_error = np.mean(np.abs(_percentage_error(actual, predicted))) * 100
        return pred_error

    def evaluate(self, individual):
        if individual is None:
            return None
        error, weights = self.get_error(individual, self.__train_set)

        if error is None:
            error = self.__invalid_fitness

        if self.test_set_file is not None:
            test_error = self.get_test_error(individual, self.__test_set, weights)

        return error, {'generation': 0, "evals": 1, "test_error": test_error, "weigths": list(weights)}


if __name__ == "__main__":
    import sge
    eval_func = EnergyPrediction(training_set_file='energy_prediction/resources/Training_Data_Spain.csv',
                                 test_set_file='energy_prediction/resources/Test_Data_Spain.csv')
    sge.evolutionary_algorithm(evaluation_function=eval_func)
    # print(eval_func.evaluate('w[0]+w[10]*(x[11])**(w[6])-w[12]-(x[13])**(w[5])*w[3]*x[11]-_log_(abs(w[2]*x[
    # 1]))*_log_('
    #                     'abs(w[12]+x[1]))'))
