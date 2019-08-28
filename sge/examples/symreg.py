import random
from math import sin, cos, tan
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


class SymbolicRegression():
    def __init__(self, function="quarticpolynomial", has_test_set=False, invalid_fitness=9999999):
        self.__train_set = []
        self.__test_set = None
        self.__number_of_variables = 1
        self.__invalid_fitness = invalid_fitness
        self.partition_rng = random.Random()
        self.function = function
        self.has_test_set = has_test_set
        self.readpolynomial()
        self.calculate_rrse_denominators()

    def calculate_rrse_denominators(self):
        self.__RRSE_train_denominator = 0
        self.__RRSE_test_denominator = 0

        train_outputs = [entry[-1] for entry in self.__train_set]
        train_output_mean = float(sum(train_outputs)) / len(train_outputs)
        self.__RRSE_train_denominator = sum([(i - train_output_mean)**2 for i in train_outputs])
        if self.__test_set:
            test_outputs = [entry[-1] for entry in self.__test_set]
            test_output_mean = float(sum(test_outputs)) / len(test_outputs)
            self.__RRSE_test_denominator = sum([(i - test_output_mean)**2 for i in test_outputs])

    def read_fit_cases(self):
        f_in = open(self.__file_problem,'r')
        data = f_in.readlines()
        f_in.close()
        fit_cases_str = [ case[:-1].split() for case in data[1:]]
        self.__train_set = [[float(elem) for elem in case] for case in fit_cases_str]
        self.__number_of_variables = len(self.__train_set[0]) - 1

    def readpolynomial(self):
        def quarticpolynomial(inp):
            return pow(inp,4) + pow(inp,3) + pow(inp,2) + inp

        def kozapolynomial(inp):
            return pow(inp,6) - (2 * pow(inp,4)) + pow(inp,2)

        def pagiepolynomial(inp1,inp2):
            return 1.0 / (1 + pow(inp1,-4.0)) + 1.0 / (1 + pow(inp2,-4))

        def keijzer6(inp):
            return sum([1.0/i for i in range(1,inp+1,1)])

        def keijzer9(inp):
            return _log_(inp + (inp**2 + 1)**0.5)

        if self.function in ["pagiepolynomial"]:
            # two variables
            xx = list(drange(-5,5.4,.4))
            yy = list(drange(-5,5.4,.4))

            function = eval(self.function)
            zz = map(function, xx, yy)
            self.__train_set=zip(xx,yy,zz)
            self.training_set_size = len(self.__train_set)
            if self.has_test_set:
                xx = list(drange(-5,5.0,.1))
                yy = list(drange(-5,5.0,.1))
                function = eval(self.function)
                zz = map(function, xx, yy)

                self.__test_set = zip(xx,yy,zz)
                self.test_set_size = len(self.__test_set)
        else:
            if self.function == "keijzer6":
                xx = list(drange(1,51,1))
            elif self.function == "keijzer9":
                xx = list(drange(0,101,1))
            else:
                xx = list(drange(-1,1.1,.1))
            function = eval(self.function)
            yy = map(function,xx)
            self.__train_set = list(zip(xx, yy))
            self.__number_of_variables = 1
            self.training_set_size = len(self.__train_set)
            if self.has_test_set:
                if self.function == "keijzer6":
                    xx = list(drange(51,121,1))
                elif self.function == "keijzer9":
                    xx = list(drange(0,101,.1))
                yy = map(function,xx)
                self.__test_set = list(zip(xx, yy))
                self.test_set_size = len(self.__test_set)

    def get_error(self, individual, dataset):
        pred_error = 0
        for fit_case in dataset:
            case_output = fit_case[-1]
            try:
                result = eval(individual, globals(), {"x": fit_case[:-1]})
                pred_error += (case_output - result)**2
            except (OverflowError, ValueError) as e:
                return self.__invalid_fitness
        return pred_error

    def evaluate(self, individual):
        error = 0.0
        test_error = 0.0
        if individual is None:
            return None

        error = self.get_error(individual, self.__train_set)
        error = _sqrt_( error /self.__RRSE_train_denominator);

        if error is None:
            error = self.__invalid_fitness

        if self.__test_set is not None:
            test_error = 0
            test_error = self.get_error(individual, self.__test_set)
            test_error = _sqrt_( test_error / float(self.__RRSE_test_denominator))

        return error, {'generation': 0, "evals": 1, "test_error": test_error}


if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func)
