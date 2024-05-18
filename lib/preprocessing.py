import numpy as np
import pandas as pd
from collections import defaultdict
import torch

def get_valid_df(df, split_date_list):
    df = df.iloc[:, :1]
    df.columns = ['c']

    # remove 0 and nan
    df = df[df!=0]
    df.dropna(inplace=True, how='any')

    # drop empty df
    if df.empty: return None

    # cal return and filter 0
    df = df.pct_change().dropna()
    df = df[df.abs()>1e-4].dropna()

    # drop empty df
    if df.empty: return None

    # demean
    df = df-df.mean()

    # add target
    df['target'] = df['c'].shift(-1)
    df.dropna(inplace=True)

    # drop short stock
    start_date, train_end_date, val_end_date = split_date_list
    if len(df.loc[start_date:train_end_date-pd.Timedelta(days=1)]) < 252*5: return None
    if len(df.loc[train_end_date:val_end_date-pd.Timedelta(days=1)]) < 252: return None
    if len(df.loc[val_end_date:]) < 252: return None

    return df.astype(np.float32)


def build_data_dict_from_df(df_list, split_date_list): # For Empirical Data
    data_dict = defaultdict(list)
    start_date, train_end_date, val_end_date = split_date_list
    df_list = [df*100 for df in df_list]

    for df in df_list:
        data_all = torch.tensor(df.to_numpy())
        data_train = torch.tensor(df.loc[start_date:train_end_date-pd.Timedelta(days=1)].to_numpy())
        data_val = torch.tensor(df.loc[train_end_date:val_end_date-pd.Timedelta(days=1)].to_numpy())
        data_test = torch.tensor(df.loc[val_end_date:].to_numpy())

        data_dict['rics'].append(df.attrs['RIC'])

        data_dict['X_train'].append(data_train[:,0].unsqueeze(1))
        data_dict['y_train'].append(data_train[:,1].unsqueeze(1))

        data_dict['X_val'].append(data_val[:,0].unsqueeze(1))
        data_dict['y_val'].append(data_val[:,1].unsqueeze(1))

        data_dict['X_test'].append(data_test[:,0].unsqueeze(1))
        data_dict['y_test'].append(data_test[:,1].unsqueeze(1))

        data_dict['X'].append(data_all[:,0].unsqueeze(1))
        data_dict['y'].append(data_all[:,1].unsqueeze(1))
    return data_dict


def build_data_dict_from_arr(arr): # For GARCH simulation
    data_dict = {}
    X = torch.tensor(arr[:, :-1, 0]).unsqueeze(-1)
    y = torch.tensor(arr[:, 1:, 0]).unsqueeze(-1)
    h = torch.tensor(arr[:, 1:, 1]).unsqueeze(-1)

    t = X.size(1)
    t_train = int(t*0.6)
    t_val = int(t*0.2)

    data_dict['X_train'] = X[:, :t_train]
    data_dict['y_train'] = y[:, :t_train]
    data_dict['h_train'] = h[:, :t_train]

    data_dict['X_val'] = X[:, t_train:t_train+t_val]
    data_dict['y_val'] = y[:, t_train:t_train+t_val]
    data_dict['h_val'] = h[:, t_train:t_train+t_val]

    data_dict['X_test'] = X[:, t_train+t_val:]
    data_dict['y_test'] = y[:, t_train+t_val:]
    data_dict['h_test'] = h[:, t_train+t_val:]

    data_dict['X'] = X
    data_dict['y'] = y
    data_dict['h'] = h

    data_dict['rics'] = [f'GARCH_{i}' for i in range(X.size(0))]

    return data_dict


