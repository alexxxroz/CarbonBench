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
        
        IGBP_CLASSES = ["CRO","CSH","CVM","DBF","DNF","EBF","ENF","GRA","MF","OSH","SAV","SNO","URB","WAT","WET","WSA"]
        KOPPEN_CLASSES = ["A","B","C","D","E"]
        KOPPEN_SHORT_CLASSES = ["Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Cfa", "Cwa", "Csa", "Csb", "Csc", "Cwb", "Cwc", 
                                "Cfb", "Cfc", "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", "Dfa", "Dfb", "Dfc", "Dfd", "ET", "EF"]
        self.igbp2id = {c:i for i,c in enumerate(IGBP_CLASSES)}
        self.koppen2id = {c:i for i,c in enumerate(KOPPEN_CLASSES)}
        
        
        df = pd.concat(hist, axis=0)
        # Fit or use provided encoders
        if encoders is None:
            self.encoders = {}
            for col in self.cat_features:
                if col == 'IGBP':
                    categories = [IGBP_CLASSES]
                elif col == 'Koppen':
                    categories = [KOPPEN_CLASSES]
                elif col == 'Koppen_short':
                    categories = [KOPPEN_SHORT_CLASSES]
                else:
                    categories = 'auto'

                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=categories)
                enc.fit(df[[col]])
                self.encoders[col] = enc
        else:
            self.encoders = encoders
        
        self.indices = self._build_indices()
    
    def _build_indices(self):
        indices = []
        self.site_data = {} 

        for site in self.hist.keys():
            df_site = self.hist[site].copy()

            cat_encoded = []
            for col in self.cat_features:
                encoded = self.encoders[col].transform(df_site[[col]])
                cat_encoded.append(encoded)
            cat_encoded = np.concatenate(cat_encoded, axis=1)

            self.site_data[site] = (df_site, cat_encoded)

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
                            indices.append((site, i))
                    else:
                        indices.append((site, i))
        return indices
    
    def get_site_indices(self, site):
        # get sample indexes by site
        return [idx for idx, (s, _) in enumerate(self.indices) if s == site]
    
    def get_sites(self):
        # get all site names in the dataset
        return list(set(s for s, _  in self.indices))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        site, i = self.indices[idx]
        df_site, cat_encoded = self.site_data[site]
        
        df_x = df_site.loc[:, ~df_site.columns.isin(
            self.targets + ['date', 'site'] + self.cat_features)]
        df_y = df_site[[col for col in self.targets if col!='NEE_VUT_USTAR50_QC']]
        
        x_window = df_x.values[i : i + self.window_size, :]
        cat_window = cat_encoded[i : i + self.window_size, :]
        
        y_target = df_y.values[i + self.window_size - self.stride : i + self.window_size, :]
        qc_w = df_site['NEE_VUT_USTAR50_QC'].values[i + self.window_size - self.stride : i + self.window_size] if self.include_qc else np.ones_like(y_target[:,0])
        
        igbp_w = df_site["IGBP"].values[i + self.window_size - self.stride : i + self.window_size]
        koppen_w = df_site["Koppen"].values[i + self.window_size - self.stride : i + self.window_size]
        igbp_w = np.array([self.igbp2id.get(v, -1) for v in igbp_w], dtype=np.int64) # -1 for unknown class
        koppen_w = np.array([self.koppen2id.get(v, -1) for v in koppen_w], dtype=np.int64) # -1 for unknown class
        
        return torch.tensor(x_window, dtype=torch.float32), \
               torch.tensor(cat_window, dtype=torch.float32), \
               torch.tensor(y_target, dtype=torch.float32), \
               torch.tensor(qc_w, dtype=torch.float32), \
               torch.from_numpy(igbp_w), \
               torch.from_numpy(koppen_w)

class SlidingWindowDatasetTAMRL(SlidingWindowDataset):
    def __init__(self, hist: dict, targets: list, include_qc: bool, QC_threshold: int=0,
                 window_size: int=30, stride: int=15, cat_features: list=['IGBP', 'Koppen', 'Koppen_short'], encoders=None):
        super().__init__(hist, targets, include_qc, QC_threshold, window_size, stride, cat_features, encoders)
    
        # Build site -> [indices] mapping
        self.site_to_indices = {}
        for idx, (site, i) in enumerate(self.indices):
            if site not in self.site_to_indices:
                self.site_to_indices[site] = []
            self.site_to_indices[site].append(idx)
    
    def __getitem__(self, idx):
        site, i = self.indices[idx]
        
        # Sample different index from same site
        candidates = [j for j in self.site_to_indices[site] if j != idx]
        support_idx = np.random.choice(candidates) if candidates else idx
        
        anchor = self._get_sample(idx)
        
        support_site, support_i = self.indices[support_idx]
        df_site, cat_encoded = self.site_data[support_site]
        df_x = df_site.loc[:, ~df_site.columns.isin(
            self.targets + ['date', 'site'] + self.cat_features)]

        x_support = torch.tensor(df_x.values[support_i : support_i + self.window_size, :], dtype=torch.float32)
        cat_support = torch.tensor(cat_encoded[support_i : support_i + self.window_size, :], dtype=torch.float32)

        return (*anchor, x_support, cat_support)
    
    def get_site_historical(self, site):
        df_site, cat_encoded = self.site_data[site]
        df_x = df_site.loc[:, ~df_site.columns.isin(
            self.targets + ['date', 'site'] + self.cat_features)]

        x_windows = []
        cat_windows = []
        for i in range(0, len(df_site) - self.window_size + 1, self.window_size):  # non-overlapping
            x_win = df_x.values[i : i + self.window_size, :]
            if not np.isnan(x_win).any():
                x_windows.append(x_win)
                cat_windows.append(cat_encoded[i : i + self.window_size, :])

        return (torch.tensor(np.stack(x_windows), dtype=torch.float32),
                torch.tensor(np.stack(cat_windows), dtype=torch.float32))
    
    def _get_sample(self, idx):
        site, i = self.indices[idx]
        df_site, cat_encoded = self.site_data[site]
        
        df_x = df_site.loc[:, ~df_site.columns.isin(
            self.targets + ['date', 'site'] + self.cat_features)]
        df_y = df_site[[col for col in self.targets if col!='NEE_VUT_USTAR50_QC']]
        
        x_window = df_x.values[i : i + self.window_size, :]
        cat_window = cat_encoded[i : i + self.window_size, :]
        
        y_target = df_y.values[i + self.window_size - self.stride : i + self.window_size, :]
        qc_w = df_site['NEE_VUT_USTAR50_QC'].values[i + self.window_size - self.stride : i + self.window_size] if self.include_qc else np.ones_like(y_target[:,0])
        
        igbp_w = df_site["IGBP"].values[i + self.window_size - self.stride : i + self.window_size]
        koppen_w = df_site["Koppen"].values[i + self.window_size - self.stride : i + self.window_size]
        igbp_w = np.array([self.igbp2id.get(v, -1) for v in igbp_w], dtype=np.int64)
        koppen_w = np.array([self.koppen2id.get(v, -1) for v in koppen_w], dtype=np.int64)
        
        return (torch.tensor(x_window, dtype=torch.float32),
                torch.tensor(cat_window, dtype=torch.float32),
                torch.tensor(y_target, dtype=torch.float32),
                torch.tensor(qc_w, dtype=torch.float32),
                torch.from_numpy(igbp_w),
                torch.from_numpy(koppen_w))
    
def historical_cache(df: pd.DataFrame, era: pd.DataFrame, mod: pd.DataFrame, x_scaler: sklearn.preprocessing._data.StandardScaler, 
                      window_size: int, cat_features: list=['IGBP', 'Koppen', 'Koppen_short']):
    """
    Precompute extra historical window for every site
    
    Why? The original data is tabular and joined as x_i, y_i. However, for temporal modeling you might need a series of inputs observed before the label was measured.
    So this function derives an set of features starting at (t_0 - window_size).
    
    """
    era_grouped = {site: g for site, g in era.groupby('site')}
    mod_grouped = {site: g for site, g in mod.groupby('site')}
    
    site_data = {}
    for site in df.site.unique():
        df_site = df[df.site == site]
        first_date = df_site['date'].min()
        window_start = first_date - timedelta(days=window_size - 1)
        
        extra_era = era_grouped.get(site, pd.DataFrame())
        extra_era = extra_era[(extra_era.date >= window_start) & (extra_era.date < first_date)]
        
        extra_mod = mod_grouped.get(site, pd.DataFrame())
        extra_mod = extra_mod[(extra_mod.date >= window_start) & (extra_mod.date < first_date)]
        
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