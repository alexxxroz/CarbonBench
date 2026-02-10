'''
This module contains util functions to process, split and visualize ground true observations.
'''

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..", "..")
DATA = os.path.join(ROOT, "data")

def load_targets(
        targets: list=['GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50', 'NEE_VUT_USTAR50'],
        qc: bool=True
    ):
    '''
    This function loads a file containing ground-true, pre-processes it, and joins Koppen climate classes,
    returning a pandas dataframe as a result.
    '''
    df = pd.read_parquet(f'{DATA}/target_fluxes.parquet')
    df = df.replace(-9999, np.nan)
    df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP, format='%Y%m%d')
    df = df.rename(columns={'TIMESTAMP': 'date'})
    df = df[df.date>=pd.to_datetime('2000-02-24')] # start date of MODIS observations

    with open(f'{DATA}/koppen_sites.json', 'r') as file:
        koppen = json.load(file)
    with open(f'{DATA}/koppen_sites_short.json', 'r') as file:
        koppen_short = json.load(file)
    df['Koppen'] = df['site'].map(koppen)
    df['Koppen_short'] = df['site'].map(koppen_short)
    
    col2keep = ['date', 'site', 'lat', 'lon', 'IGBP', 'Koppen', 'Koppen_short'] + targets
    if qc:
        col2keep += ['NEE_VUT_USTAR50_QC']
    df = df[col2keep]
    df.dropna(inplace=True)
    return df

def split_targets(
        df: pd.DataFrame, 
        split_type: str='IGBP',
        verbose: bool=True, 
        plot: bool=True,
        save_path: str='',
        **kwargs):
    '''
    This function performs constrained stratified train-test split of targets. 
    It ensures equal stratification of sites by Koppen climate class
    and IGBP class in the test set depending on the split_type (Koppen vs IGBP).
    '''
    random_state = 56 # do not change the random state, otherwise your results won't be comparable to the results of others
    
    site_meta = df.groupby('site').agg({
        'Koppen': 'first',
        'IGBP': 'first'
    }).reset_index()

    if split_type=='IGBP':
        # Step 1: Force rare (<=10 sites) IGBP classes to be equal split
        igbp_counts = site_meta.IGBP.value_counts()
        rare_igbp = igbp_counts[igbp_counts <= 10].index.tolist()

        train_sites, test_sites = [], []
        for igbp in rare_igbp:
            test_site_igbp = site_meta[site_meta.IGBP==igbp].sample(igbp_counts[igbp] // 2, random_state=random_state)
            test_sites.extend(test_site_igbp.site)
            train_site_igbp = site_meta[(site_meta.IGBP==igbp)&(~site_meta.site.isin(test_site_igbp.site))]
            train_sites.extend(train_site_igbp.site)

        # Step 2: Stratified split on remaining by IGBP
        remaining = site_meta[~site_meta.IGBP.isin(rare_igbp)]
        train_sites_str, test_sites_str = train_test_split(
            remaining.site,
            test_size=0.2,
            stratify=remaining.IGBP,
            random_state=random_state
        )

        test_sites.extend(test_sites_str)
        train_sites.extend(train_sites_str)
        
    elif split_type=='Koppen':
        train_sites, test_sites = train_test_split(
            site_meta.site,
            test_size=0.2,
            stratify=site_meta.Koppen,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown split_type={split_type}")
        
    if verbose:
        print(f"Train sites: {len(train_sites)}, Test sites: {len(test_sites)}")
        
        if split_type=='Koppen':
            test_koppen = site_meta[site_meta.site.isin(test_sites)].Koppen.value_counts(normalize=False)
            all_koppen = site_meta.Koppen.value_counts(normalize=False)
            print("\nKoppen balance:")
            print(pd.DataFrame({'overall': all_koppen, 'test': test_koppen}))
        else:
            test_igbp = site_meta[site_meta.site.isin(test_sites)].IGBP.value_counts(normalize=False)
            all_igbp = site_meta.IGBP.value_counts(normalize=False)
            print("\nIGBP balance:")
            print(pd.DataFrame({'overall': all_igbp, 'test': test_igbp}))
            
    y_train, y_test = df[df.site.isin(train_sites)], df[df.site.isin(test_sites)] 
    
    if plot:
        plot_sites(y_train.copy(), y_test.copy(), split_type, save_path)

    return y_train, y_test
        
def plot_sites(
        train: pd.DataFrame, 
        test: pd.DataFrame, 
        split_type: str='IGBP', 
        save_path: str=''
    ):
    '''
    This function takes train and test dataframes and plots all the sites on the same map.
    '''
    train['type'] = 'train'
    test['type'] = 'test'
    df = pd.concat([train,test])
    
    site_colors = {
        'test': '#FF0000',  
        'train': '#2196F3',  
    }

    fig = plt.figure(figsize=(16, 10), facecolor='#0a1128') 
    ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine())

    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor='#001f54', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#4a4a62', edgecolor='#2d2d44', linewidth=0.3, zorder=1)

    for idx, group in df.groupby('site'):
        lat, lon, label = group.lat.values[0], group.lon.values[0], group.type.values[0]
        ax.scatter(lon, lat, c=site_colors[label], s=75, edgecolors='white', 
                   linewidths=2, zorder=5, transform=ccrs.PlateCarree(), alpha=0.95)
        ax.scatter(lon, lat, c=site_colors[label], s=200, alpha=0.2, 
                   transform=ccrs.PlateCarree(), zorder=4)

    legend_elements = [
        Patch(facecolor=site_colors['train'], edgecolor='white', label=f'Train ({train.site.nunique()} sites)'),
        Patch(facecolor=site_colors['test'], edgecolor='white', label=f'Test ({test.site.nunique()} sites)'),
    ]

    ax.legend(handles=legend_elements, loc='lower left', frameon=False, 
              fontsize=11, labelcolor='white')

    ax.set_facecolor('#001f54')
    ax.spines['geo'].set_visible(False)
    ax.axis('off')
    ax.set_title(f'CarbonBench (split={split_type}): train vs test sites', color='white', fontsize=22)

    plt.tight_layout(pad=0)
    if len(save_path) > 0:
        plt.savefig(os.path.join(save_path, 'sites_map.png'), bbox_inches='tight', edgecolor='none') 
    plt.show()
    
def plot_site_ts(
        df: pd.DataFrame, 
        targets: list, 
        include_qc: bool=True, 
        qc_threshold: int=0.75, 
        site_name: str='random', 
        save_path: str=''
    ):
    '''
    This function creates a time series plot for targets from a given or randomly selected site. 
    If NEE_VUT_USTAR50_QC is included, the areas of high and low data quality are highlighted.
    '''
    koppen_map = {
        'A': 'Tropical',
        'B': 'Arid',
        'C': 'Temperate',
        'D': 'Continental',
        'E': 'Polar'
    }
    if site_name=='random':
        site_name = np.random.choice(df.site.unique(), 1).item()
    
    sub_df = df[df.site==site_name].set_index('date')
    fig, ax = plt.subplots(figsize=(16,9))
    sub_df[targets].plot(ax=ax, color=["#543005", "#bf812d", '#003c30'], lw=0.5)
    ymin = sub_df[targets].min().min()
    ymax = sub_df[targets].max().max()
    
    if include_qc:
        # Low QC shading (<0.75)
        ax.fill_between(
            sub_df.index,
            ymin, ymax,
            where=sub_df['NEE_VUT_USTAR50_QC'] < qc_threshold,
            color='red',
            alpha=0.2,
            interpolate=True
        )

        # High QC shading (>=0.75)
        ax.fill_between(
            sub_df.index,
            ymin, ymax,
            where=sub_df['NEE_VUT_USTAR50_QC'] >= qc_threshold,
            color='green',
            alpha=0.1,
            interpolate=True
        )
    ax.grid(0.5, ls='--', color='grey')
    ax.set_title(f"Site: {site_name}, IGBP: {sub_df.IGBP.unique()[0]}, Koppen: {koppen_map[sub_df.Koppen.unique()[0]]}")
    plt.tight_layout()
    if len(save_path) > 0:
        plt.savefig(os.path.join(save_path, f'{site_name}_ts.png')) 
    plt.show()
        
    