import numpy as np
import pandas as pd

def sliding_window(df: pd.DataFrame, targets: list, include_qc: bool, window_size: int=30, stride: int=15):
    site_ids = df.site.unique()
    X, y = [], []
    if include_qc:
        targets += ['NEE_VUT_USTAR50_QC']
        
    for site in site_ids:
        df_site = df[df.site==site].copy()
        df_site = df_site.set_index('date').resample('D').asfreq().reset_index()
        df_site[mod_bands] = df_site[mod_bands].interpolate(limit_direction='both')

        df_x, df_y = df_site[targets], df_site.loc[:, ~df_site.columns.isin(targets)]
    
    
        for i in range(0, len(df_site) - window_size + 1, stride):
            x_window = df_x.values[i : i + window_size, :]
            y_target = df_y.values[i + window_size - stride : i + window_size, :]
            
            if pd.isna(x_window).any() or pd.isna(y_target).any(): 
                continue
                
            X.append(x_window)
            y.append(y_target)
    X, y = np.array(X), np.array(y)
    return X, y
    
def tabular():
    return X_train, y_train

mod_bands = [f'sur_refl_b0{i}' for i in range(1,8)] + ['SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth', 'clouds']