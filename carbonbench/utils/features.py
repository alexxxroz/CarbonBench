import json
import numpy as np
import pandas as pd

def load_modis():
    df = pd.read_parquet('./data/MOD09GA.parquet')
    df.TIMESTAMP = pd.to_datetime(df.date)
    return df

def load_era(set_type: str='standard'):
    with open('./data/feature_sets.json', 'r') as file:
        features = json.load(file)
    col2keep = ['date', 'site'] + features[set_type]
    
    df = pd.read_parquet('./data/ERA5.parquet', columns=columns)
    df.TIMESTAMP = pd.to_datetime(df.date)
    return df