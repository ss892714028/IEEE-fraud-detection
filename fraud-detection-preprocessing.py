import pandas as pd
import sklearn
import datetime
import time


def dataloader():
    train_identity = pd.read_csv(r'.\ieee-fraud-detection\train_identity.csv')
    train_transaction = pd.read_csv(r'.\ieee-fraud-detection\train_transaction.csv')
    test_identity = pd.read_csv(r'.\ieee-fraud-detection\test_identity.csv')
    test_transaction = pd.read_csv(r'.\ieee-fraud-detection\test_transaction.csv')
    train = pd.merge(train_transaction, train_identity, on = 'TransactionID', how = 'left')
    test = pd.merge(test_transaction, test_identity, on = 'TransactionID', how = 'left')
    train_label = train['isFraud']
    train = train.drop(columns = ['isFraud'])
    all_data = pd.concat([train, test], axis = 0)
    return all_data, train, test, train_label


def get_feature_type(df):
    # make a dictionary to store all feature type
    # we know int and float are continuous O is categorical
    # cont for continuous
    # cat for catagorical
    dtype_dict = {}
    for i in df.columns.tolist():
        dtype_dict[i] = train[i].dtype

    feature_type = {}
    for i in dtype_dict.keys():
        if dtype_dict[i] == 'int64':
            feature_type[i] = 'cont'
        if dtype_dict[i] == 'float64':
            feature_type[i] = 'cont'
        if dtype_dict[i] == 'O':
            feature_type[i] = 'cat'
    return feature_type


def find_unique_value(df, feature_type):
    feature_unique_value = {}
    for i in df.columns.tolist():
        if feature_type[i] == 'cat':
            feature_unique_value[i] = len(set(all_data[i].unique()))
    return feature_unique_value


def reduce_f_dim(col, critical_value):
    temp = col.value_counts().to_dict()
    total = sum(temp.values())
    new_col = []
    for keys in col:
        if temp[keys] / total > critical_value:
            new_col.append(keys)
        else:
            new_col.append('etc')
    return new_col


def categorical_feature_reduce_dim(df, feature_unique_value,
                                   critical_feature_num, critical_value):
    for i in feature_unique_value.keys():
        if feature_unique_value[i] > critical_feature_num:
            col = df[i]
            fill_value = 'null'
            col = col.fillna(fill_value)
            col = reduce_f_dim(col, critical_value)
            df[i] = col
    return df


def add_date(df):
    startdate = datetime.datetime.strptime('20190101', '%Y%m%d')
    df['Date'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    df['Week'] = df['Date'].dt.dayofweek
    df['days'] = df['Date'].dt.day
    df['hours'] = df['Date'].dt.hour
    df = df.drop(columns=['Date'])
    return df


# find percentage of non null value of each feature
def find_null_percentage(df):
    null = {}
    for i in df.columns.tolist():
        null[i] = 1 - df[i].isnull().value_counts()[False]/df.shape[0]
    return null


def remove_na_col(df, critical_value, null):
    columns_to_drop = []
    for i in null.keys():
        if null[i] > critical_value:
            columns_to_drop.append(i)
    df = df.drop(columns=columns_to_drop)

    return df


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


def fill_numeric_na(df, method, feature_type):
    for i in df.columns.tolist():
        if feature_type[i] == 'cont':
            df[i] = fill_null(df[i], method)
    return df


if __name__ == '__main__':
    t = time.time()

    print('loading data...')
    all_data, train, test, train_label = dataloader()

    # find type of each feature (cat:categorical or cont:continuous)
    # in dict format
    print('making feature_type dict...')
    feature_type_dict = get_feature_type(all_data)

    # find number of unique values in each categorical features
    # in dict format
    print('making unique_value_dict...')
    unique_value_dict = find_unique_value(all_data,feature_type_dict)

    # find percentage of null value in each feature
    # in dict format
    print('finding null_percentage...')
    null_dict = find_null_percentage(all_data)

    print('adding datetime features...')
    train = add_date(train)

    print('reducing categorical feature dimension...')
    train = categorical_feature_reduce_dim(df=train, feature_unique_value=unique_value_dict,
                                           critical_feature_num=8,critical_value=0.05)
    print('deleting na columns...')
    train=remove_na_col(train, 0.25, null_dict)

    print('filling na for numeric columns...')
    train = fill_numeric_na(train,'avg',feature_type_dict)

    print('time ultilized {}'.format(time.time() - t))
