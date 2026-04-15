from numpy import cos, sin, random
import torch
import pandas as pd
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.utilities.pytorchtest import *


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


class RMSE():
    def __init__(self, train_set, test_set):
        self.criterion = torch.nn.MSELoss()
        
    def calculate(self, predicted, dataset):
        pred_error = torch.sqrt(self.criterion(predicted, dataset.target))
        return pred_error


class RRSE():

    def __init__(self, train_set, test_set):
        self.criterion = torch.nn.MSELoss(reduction='sum')
        train_output_mean = torch.mean(train_set.target)
        self.__RRSE_train_denominator = self.criterion(train_set.target, train_output_mean)
        if test_set:
            test_output_mean = torch.mean(test_set.target)
            self.__RRSE_test_denominator = self.criterion(test_set.target, test_output_mean)

    def calculate(self, predicted, dataset):
        pred_error = self.criterion(predicted, dataset.target)

        if dataset.type == "test":
            error = torch.sqrt(torch.div(pred_error, self.__RRSE_test_denominator))
        else:
            error = torch.sqrt(torch.div(pred_error, self.__RRSE_train_denominator))
        return error
    

class Dataset():
    def __init__(self, dataset, type):
        dataset = torch.tensor(dataset, dtype=torch.float32, device=cur_dev)
        self.values = dataset[:, :-1]
        self.target = dataset[:, -1]
        self.type = type


class SymbolicRegression():
    def __init__(self, function="nguyen5polynomial", has_test_set=False, invalid_fitness=9999999):
        self.__train_set = []
        self.__test_set = None
        self.__invalid_fitness = invalid_fitness
        self.partition_rng = random.uniform()
        self.function = function
        self.has_test_set = has_test_set
        self.readpolynomial()
        self.__loss_function = RRSE(self.__train_set, self.__test_set)

    def read_fit_cases(self):
        f_in = open(self.__file_problem,'r')
        data = f_in.readlines()
        f_in.close()
        fit_cases_str = [ case[:-1].split() for case in data[1:]]
        self.__train_set = Dataset([[float(elem) for elem in case] for case in fit_cases_str], "train")

    def readpolynomial(self):
        def quarticpolynomial(inp):
            return pow(inp,4) + pow(inp,3) + pow(inp,2) + inp

        def kozapolynomial(inp):
            return pow(inp,6) - (2 * pow(inp,4)) + pow(inp,2)

        def pagiepolynomial(inp1,inp2):
            return 1.0 / (1 + pow(inp1,-4.0)) + 1.0 / (1 + pow(inp2,-4))
        
        def nguyen5polynomial(inp):
            return node_sub(node_mul(node_sin(inp**2),node_cos(inp)), 1)

        def keijzer6(inp):
            return sum([1.0/i for i in range(1,inp+1,1)])

        def keijzer9(inp):
            return _log_(inp + (inp**2 + 1)**0.5)

        if self.function in ["pagiepolynomial"]:
            function = eval(self.function)
            # two variables
            l = []
            for xx in drange(-5,5.4,0.4):
                for yy in drange(-5,5.4,0.4):
                    zz = pagiepolynomial(xx,yy)
                    l.append([xx,yy,zz])

            self.training_set_size = len(l)
            self.__train_set=Dataset(l, "train")
            if self.has_test_set:
                xx = list(drange(-5,5.0,.1))
                yy = list(drange(-5,5.0,.1))
                function = eval(self.function)
                zz = map(function, xx, yy)

                self.__test_set = Dataset([xx,yy,zz], "test")
                self.test_set_size = len(self.__test_set)
        elif self.function in ["quarticpolynomial"]:
            function = eval(self.function)
            l = []
            for xx in drange(-1,1.1,0.1):
                yy = quarticpolynomial(xx)
                l.append([xx,yy])

            self.__train_set = Dataset(l, "train")
            self.training_set_size = len(l)
            if self.has_test_set:
                xx = list(drange(-1,1.1,0.1))
                function = eval(self.function)
                yy = map(function, xx)

                self.__test_set = [xx,yy]
                self.test_set_size = len(self.__test_set)
        elif self.function in ["nguyen5polynomial"]:
            function = eval(self.function)
            l = []
            for xx in drange(-1,1.1,0.1):
                yy = nguyen5polynomial(xx)
                l.append([xx,yy])

            self.__train_set = Dataset(l, "train")
            self.training_set_size = len(l)
            if self.has_test_set:
                xx = list(drange(-1,1.1,0.1))
                function = eval(self.function)
                yy = map(function, xx)

                self.__test_set = [xx,yy]
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
            l = list(zip(xx, yy))
            self.__train_set = Dataset(l, "train")
            self.__number_of_variables = 1
            self.training_set_size = len(l)
            if self.has_test_set:
                if self.function == "keijzer6":
                    xx = list(drange(51,121,1))
                elif self.function == "keijzer9":
                    xx = list(drange(0,101,.1))
                yy = map(function,xx)
                self.__test_set = Dataset([xx, yy],"test")
                self.test_set_size = len(self.__test_set)

    def get_error(self, dataset):
        pred_error = 0.0
        try:
            predicted = torch.vmap(
                corre,
                0)(dataset.values)

            pred_error = self.__loss_function.calculate(predicted, dataset)
            if pred_error.isnan() or pred_error.isinf():
                 pred_error = torch.tensor(self.__invalid_fitness, device=cur_dev)

        except (OverflowError, ValueError) as e:
            return torch.tensor(self.__invalid_fitness, device=cur_dev)

        return pred_error

    def evaluate(self, individual):
        error = 0.0
        test_error = 0.0
        if individual is None:
            return None
        
        code = compile('def corre(x): \n\t return %s' % individual, '<string>', 'exec')
        exec(code, globals())

        error = self.get_error(self.__train_set)

        if self.__test_set:
            test_error = self.get_error(self.__test_set)
            return (float(error.cpu().data.numpy()), {'generation': 0, "evals": 1, "test_error": float(test_error.cpu().data.numpy())})

        return (float(error.cpu().data.numpy()), {'generation': 0, "evals": 1, "test_error": test_error})


if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func, parameters_file="parameters/standard.yml")
