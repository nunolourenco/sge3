import pandas as pd

from fairness.datasets import Dataset
from fairness.utils import Encoding


class CompasDataset(Dataset):
    def __init__(self, encoding, name='compas', protected='race', columns_to_remove=None, subset='train'):
        classes = {0: 'recidivist', 1: 'not-recidivist'}

        super().__init__(name=name, protected=protected, target='no-recid', classes=classes, encoding=encoding)

        self.load_dataset(subset=subset)
        self.features_including_protected()
        self.features_excluding_protected()

        if columns_to_remove:
            self.remove_columns(columns_to_remove)

    def load_dataset(self, subset, convert=True):
        if self.encoding is Encoding.ONE_HOT:
            self.data = pd.read_csv(filepath_or_buffer=self.DATA_PATH + 'compas-{}-bindata.csv'.format(subset),
                                    header=0, index_col=0)
        elif self.encoding is Encoding.INTEGER:
            self.data = pd.read_csv(filepath_or_buffer=self.DATA_PATH + 'compas-{}-data.csv'.format(subset),
                                    header=0, index_col=0)
        else:
            raise ValueError('[ERROR] unsupported encoding')

        self.columns = self.data.columns.tolist()

        if convert:
            self.convert_to_default_format()
