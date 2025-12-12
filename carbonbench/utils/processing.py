import numpy as np
import pandas as pd
from datetime import timedelta
import torch
from torch.utils.data import Dataset, DataLoader

import sklearn
from sklearn.preprocessing import OneHotEncoder

class SlidingWindowDataset(Dataset):
    def __init__(self, hist: dict, targets: list, include_qc: bool, QC_threshold: int=0,
                 window_size: int=30, stride: int=15, cat_features: list=['IGBP', 'Koppen', 'Koppen_short'], encoders=None):
        self.hist = hist
        self.targets = targets + (['NEE_VUT_USTAR50_QC'] if include_qc else [])
        self.window_size = window_size
        self.stride = stride
        self.cat_features = cat_features
        self.include_qc = include_qc
        self.QC_threshold = QC_threshold
        
        df = pd.concat(hist, axis=0)
        # Fit or use provided encoders
        if encoders is None:
            self.encoders = {}
            for col in self.cat_features:
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(df[[col]])
                self.encoders[col] = enc
        else:
            self.encoders = encoders
        
        self.indices = self._build_indices()
        
    def _build_indices(self):
        indices = []
        for site in self.hist.keys():
            df_site = self.hist[site].copy()

            # Encode categorical features
            cat_encoded = []
            for col in self.cat_features:
                encoded = self.encoders[col].transform(df_site[[col]])
                cat_encoded.append(encoded)
            cat_encoded = np.concatenate(cat_encoded, axis=1)
            
            for i in range(0, len(df_site) - self.window_size + 1, self.stride):
                # Check features in window
                feature_cols = ~df_site.columns.isin(self.targets)
                x_window = df_site.iloc[i:i + self.window_size].loc[:, feature_cols]

                # Check targets for the prediction window
                target_start = i + self.window_size - self.stride
                target_end = i + self.window_size
                y_target = df_site.iloc[target_start:target_end][self.targets]

                if not x_window.isna().any().any() and not y_target.isna().any().any():
                    if self.include_qc:
                        if (y_target['NEE_VUT_USTAR50_QC'] >= self.QC_threshold).all(): # extra filtering by QC
                            indices.append((site, i, df_site, cat_encoded))
                    else:
                        indices.append((site, i, df_site, cat_encoded))
        return indices
    
    def get_site_indices(self, site):
        # get sample indexes by site
        return [idx for idx, (s, _, _, _) in enumerate(self.indices) if s == site]
    
    def get_sites(self):
        # get all site names in the dataset
        return list(set(s for s, _, _, _ in self.indices))

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        site, i, df_site, cat_encoded = self.indices[idx]
        df_x = df_site.loc[:, ~df_site.columns.isin(
            self.targets + ['date', 'site'] + self.cat_features)]
        df_y = df_site[[col for col in self.targets if col!='NEE_VUT_USTAR50_QC']]
        
        x_window = df_x.values[i : i + self.window_size, :]
        cat_window = cat_encoded[i : i + self.window_size, :]
        
        y_target = df_y.values[i + self.window_size - self.stride : i + self.window_size, :]
        
        return torch.tensor(x_window, dtype=torch.float32), \
               torch.tensor(cat_window, dtype=torch.float32), \
               torch.tensor(y_target, dtype=torch.float32)

def historical_cache(df: pd.DataFrame, era: pd.DataFrame, mod: pd.DataFrame, x_scaler: sklearn.preprocessing._data.StandardScaler, 
                      window_size: int, cat_features: list=['IGBP', 'Koppen', 'Koppen_short']):
    """Precompute extra historical window for every site"""
    site_data = {}
    for site in df.site.unique(): 
        df_site = df[df.site==site].copy()
        first_date = df_site['date'].min()
        window_start = first_date - timedelta(days=window_size-1)
        
        extra_era = era[(era.site==site) & (era.date >= window_start) & (era.date < first_date)]
        extra_mod = mod[(mod.site==site) & (mod.date >= window_start) & (mod.date < first_date)]
        extra = pd.merge(extra_era, extra_mod, on=['site', 'date'], how='outer')
        extra['lat'], extra['lon'] = df_site['lat'].unique().item(), df_site['lon'].unique().item()
        
        extra[x_scaler.feature_names_in_] = x_scaler.transform(extra[x_scaler.feature_names_in_])
        
        df_extended = pd.concat([extra, df_site]).sort_values('date')
        df_extended = df_extended.set_index('date').resample('D').asfreq().reset_index()
        df_extended[mod_bands] = df_extended[mod_bands].interpolate(limit_direction='both')
        df_extended[cat_features] = df_extended[cat_features].bfill()
        site_data[site] = df_extended
    return site_data
    
def tabular(df: pd.DataFrame, targets: list, include_qc: bool=True, QC_threshold: int=0, cat_features: list=['IGBP', 'Koppen', 'Koppen_short']):
    dfs = []
    for site in df.site.unique():
        df_site = df[df.site==site].set_index('date').resample('D').asfreq().reset_index()
        df_site[mod_bands] = df_site[mod_bands].interpolate(limit_direction='both')
        df_site = df_site.dropna(axis=0)
        if include_qc:
            df_site = df_site[df_site.NEE_VUT_USTAR50_QC>=QC_threshold]
        dfs.append(df_site)
    df = pd.concat(dfs)
    
    X = df.loc[:, ~df.columns.isin(targets + ['date', 'NEE_VUT_USTAR50_QC'])]
    X[cat_features] = X[cat_features].astype('category')
    y = df[targets]
    if include_qc:
        y_qc = df['NEE_VUT_USTAR50_QC']
        return X, y, y_qc
    else:
        return X, y

mod_bands = [f'sur_refl_b0{i}' for i in range(1,8)] + ['SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth', 'clouds']