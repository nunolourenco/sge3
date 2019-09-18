from fairness.datasets import AdultIncomeDataset, GermanCreditDataset, CompasDataset
from fairness.utils import Encoding

if __name__ == '__main__':
    DATASETS = [
        AdultIncomeDataset(encoding=Encoding.ONE_HOT),
        GermanCreditDataset(encoding=Encoding.ONE_HOT),
        CompasDataset(encoding=Encoding.ONE_HOT)
    ]

    SEED = 1234
    
    PATH = ''

    for dataset in DATASETS:
        dataset.partition_dataset(path=PATH, splits=30, test_size=0.20, seed=SEED)
