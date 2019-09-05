import pandas as pd


def find_null_percent(column):
    assert type(column) == type(pd.DataFrame([1]))

    value_count = column.isnull().value_counts()
    return value_count[True]/(value_count[True] + value_count[False])


def fill_null(column, value):
    """

    :param column: pandas Dataframe column
    :param value: 'avg', 'med', or float number
    :return: column with null entry filled
    """
    assert type(column) == type(pd.DataFrame([1]))

    if value not in ['avg', 'med']:
        print("Fill in custom value {} for null entries.".format(value))
        return column.fillna(value)
    elif value == 'avg':
        try:
            m = column.mean()
            print("Fill in average value {} for null entries.".format(m))
            return column.fillna(m)
        except TypeError:
            print("TypeError: Average method can only be used for numerical numbers")
            raise
    elif value == 'med':
        try:
            m = column.median()
            print("Fill in median value {} for null entries.".format(m))
            return column.fillna(m)
        except TypeError:
            print("TypeError: Median method can only be used for numerical numbers")
            raise






