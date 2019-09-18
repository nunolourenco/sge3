import pandas as pd

from fairness.datasets import Dataset
from fairness.utils import Encoding


class GermanCreditDataset(Dataset):
    def __init__(self, encoding, name='german-credit', protected='age', columns_to_remove=None, subset='train'):
        classes = {0: 'bad-credit', 1: 'good-credit'}

        super().__init__(name=name, protected=protected, target='credit', classes=classes, encoding=encoding)

        self.load_dataset(subset=subset)
        self.features_including_protected()
        self.features_excluding_protected()

        if columns_to_remove:
            self.remove_columns(columns_to_remove)

    def load_dataset(self, subset, convert=True):
        if self.encoding is Encoding.ONE_HOT:
            self.data = pd.read_csv(filepath_or_buffer=self.DATA_PATH + 'german-credit-{}-bindata.csv'.format(subset),
                                    header=0, index_col=0)
        elif self.encoding is Encoding.INTEGER:
            self.data = pd.read_csv(filepath_or_buffer=self.DATA_PATH + 'german-credit-{}-data.csv'.format(subset),
                                    header=0, index_col=0)
        else:
            raise NotImplementedError()

        self.columns = self.data.columns.tolist()

        if convert:
            self.convert_to_default_format()
