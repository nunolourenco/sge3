import pandas as pd


def discretize_numeric(column, nbins=4):
    filtered_column = column.dropna()

    nsamples = filtered_column.shape[0]

    sorted_column = filtered_column.sort_values(ascending=True)

    thresholds = [sorted_column.iloc[int((i + 1) * nsamples / nbins)] for i in range(nbins - 1)]
    thresholds.append(sorted_column.iloc[-1])

    series = pd.Series(index=column.index)

    for t in thresholds:
        indices = column[(column <= t) & series.isna() & (~column.isna())].index
        series.loc[indices] = 'bin' + str(t)

    temp_categories = sorted(set(thresholds))
    categories = ['bin' + str(value) for value in temp_categories]

    return series, categories


def merge_low_frequency_bins(column, min_samples=50):
    counts = column.value_counts()

    low_freq_bins = [label for label, count in counts.items() if count <= min_samples]

    if len(low_freq_bins) > 1:
        indices = column[column.isin(low_freq_bins)].index

        if indices.shape[0] > 0:
            column = column.cat.add_categories(['Pool'])
            column.loc[indices] = 'Pool'

    column = column.cat.remove_unused_categories()

    return column


def convert_to_one_hot_encoding(df, categories):
    df_binary = pd.DataFrame()
    num_categories_per_attribute = df.nunique()

    for name, num_categories in num_categories_per_attribute.items():
        if num_categories > 2:
            temp_df = pd.get_dummies(data=df[name])
            sorted_columns = sorted(temp_df.columns)
            temp_df = temp_df.reindex(labels=sorted_columns, axis=1, copy=False)
            binary_columns = dict()
            for col in temp_df.columns:
                binary_columns[col] = name + '_' + str(categories[name][col])
            temp_df.rename(binary_columns, axis=1, inplace=True)
            df_binary[temp_df.columns] = temp_df.copy()
        else:
            df_binary[name] = df[name].copy()

    return df_binary
