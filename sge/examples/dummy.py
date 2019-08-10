import sge

class Dummy:

    def evaluate(self, ind):
        return 1, {}


if __name__ == '__main__':
    sge.evolutionary_algorithm(evaluation_function=Dummy())
