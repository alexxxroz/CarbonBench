'''
This module contains util functions to perfrom loading, joining and processing of the features.
'''

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..", "..")
DATA = os.path.join(ROOT, "data")

def load_modis():
    '''
    Loads MODIS observations as a pandas dataframe.
    '''
    df = pd.read_parquet(f'{DATA}/MOD09GA.parquet')
    df.date = pd.to_datetime(df.date)
    return df

def load_era(
        set_type: str='standard'
    ):
    '''
    Loads ERA5 Land feature set -- [minimal, standard, full] -- and returns as a dataframe.
    '''
    with open(f'{DATA}/feature_sets.json', 'r') as file:
        features = json.load(file)
    col2keep = ['date', 'site'] + features[set_type]
    
    df = pd.read_parquet(f'{DATA}/ERA5.parquet', columns=col2keep)
    df.date = pd.to_datetime(df.date)
    return df

def join_features(
        y_train: pd.DataFrame, 
        y_test: pd.DataFrame, 
        modis: pd.DataFrame, 
        era: pd.DataFrame, 
        val_ratio: float=0.2, 
        scale: bool=True
    ):
    '''
    Performs left joins of features to the targets based on date and site, further splitting
    train into train and val (for early stop) and normalizes the data.
    '''
    df_train = y_train.merge(modis, how='left', on=['date', 'site'], sort=False)
    df_train = df_train.merge(era, how='left', on=['date','site'], sort=False)
    
    df_test = y_test.merge(modis, how='left', on=['date', 'site'], sort=False)
    df_test = df_test.merge(era, how='left', on=['date','site'], sort=False)
    
    # preparing validation df
    train, val = [], []
    for site in df_train.site.unique():
        site_df = df_train[df_train.site==site]
        split_idx = int(len(site_df) * (1 - val_ratio))
        train.append(site_df.iloc[:split_idx])
        val.append(site_df.iloc[split_idx:])

    df_train = pd.concat(train).reset_index(drop=True)
    df_val = pd.concat(val).reset_index(drop=True)

    if scale:
        features = ['lat', 'lon'] + [col for col in era.columns if col not in ['date', 'site']] + [col for col in modis.columns if col not in ['date', 'site']] 
        targets = [col for col in y_train.columns if 'USTAR50' in col and 'QC' not in col]
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        df_train[features] = x_scaler.fit_transform(df_train[features])
        df_val[features] = x_scaler.transform(df_val[features])
        df_test[features]  = x_scaler.transform(df_test[features])

        df_train[targets] = y_scaler.fit_transform(df_train[targets])
        df_val[targets] = y_scaler.transform(df_val[targets])
        df_test[targets]  = y_scaler.transform(df_test[targets])
        return df_train, df_val, df_test, x_scaler, y_scaler
    else:
        return df_train, df_test
    
def plot_feature_heatmap(
        df: pd.DataFrame, 
        cat_features: list=['IGBP', 'Koppen', 'Koppen_short'], 
        save_path: str='', 
        figsize: tuple=(25,25)
    ):
    '''
    Creates a heatmap of the existing continious variables.
    Warning: the image can explode when used with the Full feature set of ERA5 Land.
    '''
    plt.rcParams.update({'font.size': 14, 'font.family':'monospace', 'figure.figsize': figsize})
    sns.heatmap(df.drop(['date', 'site'] + cat_features, axis=1).corr(), fmt=".2f", annot=True, cmap="BrBG")
    if len(save_path) > 0:
        plt.savefig(os.path.join(save_path, f"featrue_heatmap.png"))
    plt.show()