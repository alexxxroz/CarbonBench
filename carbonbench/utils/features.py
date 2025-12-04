import json
import numpy as np
import pandas as pd

def load_modis():
    df = pd.read_parquet('./data/MOD09GA.parquet')
    df.date = pd.to_datetime(df.date)
    return df

def load_era(set_type: str='standard'):
    with open('./data/feature_sets.json', 'r') as file:
        features = json.load(file)
    col2keep = ['date', 'site'] + features[set_type]
    
    df = pd.read_parquet('./data/ERA5.parquet', columns=col2keep)
    df.date = pd.to_datetime(df.date)
    return df

def join_features(y: pd.DataFrame, modis: pd.DataFrame, era: pd.DataFrame):
    df = y.merge(modis, how='left', on=['date', 'site'], sort=False)
    df = df.merge(era, how='left', on=['date','site'], sort=False)
    return df