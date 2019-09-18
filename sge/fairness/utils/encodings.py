import pandas as pd

from enum import Enum, unique


@unique
class Encoding(Enum):
    ONE_HOT = 'bindata'
    INTEGER = 'data'


def convert_to_one_hot_encoding(series: pd.Series, categories):
    df_binary = pd.DataFrame()
    num_categories = series.nunique()

    if num_categories > 2:
        temp_df = pd.get_dummies(data=series)
        sorted_columns = sorted(temp_df.columns)
        temp_df = temp_df.reindex(labels=sorted_columns, axis=1, copy=False)
        binary_columns = dict()
        for col in temp_df.columns:
            binary_columns[col] = series.name + '_' + categories[col]
        temp_df.rename(binary_columns, axis=1, inplace=True)
        df_binary[temp_df.columns] = temp_df.copy()
    else:
        df_binary = pd.DataFrame(series, index=series.index)

    return df_binary
