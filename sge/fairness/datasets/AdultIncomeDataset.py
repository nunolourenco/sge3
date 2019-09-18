import pandas as pd

from fairness.datasets import Dataset
from fairness.utils import Encoding


class AdultIncomeDataset(Dataset):
    def __init__(self, encoding, name='adult-income', protected='sex', columns_to_remove=None, subset='train'):
        classes = {0: 'low-income', 1: 'high-income'}

        super().__init__(name=name, protected=protected, target='income', classes=classes, encoding=encoding)

        self.load_dataset(subset=subset)
        self.features_including_protected()
        self.features_excluding_protected()

        if columns_to_remove:
            self.remove_columns(columns_to_remove)

    def load_dataset(self, subset):
        if self.encoding is Encoding.ONE_HOT:
            self.data = pd.read_csv(filepath_or_buffer=self.DATA_PATH + 'adult-income-{}-bindata.csv'.format(subset),
                                    header=0)
        elif self.encoding is Encoding.INTEGER:
            self.data = pd.read_csv(filepath_or_buffer=self.DATA_PATH + 'adult-income-{}-data.csv'.format(subset),
                                    header=0)
        else:
            raise NotImplementedError()

        self.columns = self.data.columns.tolist()
