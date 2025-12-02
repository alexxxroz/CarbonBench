'''
    Run only if you want to reproduce the whole benchmark from scratch!
    The script reads the results of extractFluxes.py execution and joins them with meta information such as latitude, longitude and IGBP type.
    The script save the final parquet file with targets.
'''

import yaml
import json
from time import time
from os import listdir
from os.path import join

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

config_fname = '../config.yaml'
with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)
    
main_path = config['flux_path']['main_path']

df = pd.read_parquet(f'{main_path}/all_fluxes.parquet') 

'''Load site meta info'''
flux = pd.read_csv('../data/FLUXNET2015_Metadata.csv')
flux['SITE_ID'] = 'FLX_' + flux['SITE_ID']
ameri = pd.read_csv('../data/AmeriFlux_Metadata.tsv', sep='\t')
ameri['Site ID'] = 'AMF_' + ameri['Site ID']
icos = pd.read_csv('../data/ICOS2025_Metadata.csv')

'''Extracting JapanFlux metadata'''
jap_path = '/media/aleksei/303AFBEF3AFBB048/FLUX/JapanFlux2024'
jap_files = listdir(jap_path)
jap_dir = [x for x in listdir(jap_path) if (x.split('.')[-1]!='zip')]
data_jap = {x: [] for x in ['site', 'lat', 'lon', 'IGBP']}
for folder in jap_dir:
    try:
        csv = [x for x in listdir(join(jap_path, folder, folder.split('_')[0], 'DATA','BDAM')) if 'General' in x][0]
        d = pd.read_excel(join(jap_path, folder, folder.split('_')[0], 'DATA','BDAM', csv))
    except Exception as e:
        csv = [x for x in listdir(join(jap_path, folder, folder.split('_')[0], 'DATA','BADM')) if 'General' in x][0]
        d = pd.read_excel(join(jap_path, folder, folder.split('_')[0], 'DATA','BADM', csv))
    d.columns = [x.replace(" ", "") for x in d.columns]
    try:
        lat = d[d.Variable=='LOCATION_LAT']['dataValue1'].item()
        lon = d[d.Variable=='LOCATION_LONG']['dataValue1'].item()
        IGBP = d[d.Variable=='IGBP']['dataValue1'].item()[:3]
    except Exception as e:
        try:
            lat = d[d.Variable=='LOCATION_LAT']['dataValue'].item()
            lon = d[d.Variable=='LOCATION_LONG']['dataValue'].item()
            IGBP = d[d.Variable=='IGBP']['dataValue'].item()[:3]
        except Exception as e2:
            lat = d[d.Description=='LOCATION_LAT']['dataValue1'].item()
            lon = d[d.Description=='LOCATION_LONG']['dataValue1'].item()
            IGBP = d[d.Description=='IGBP']['dataValue1'].item()[:3]
    data_jap['lat'].append(lat)
    data_jap['lon'].append(lon)
    data_jap['IGBP'].append(IGBP)
    site = 'JPX_' + csv.split('_')[0]
    site = site[:10] if len(site)>10 else site
    data_jap['site'].append(site)
data_jap = pd.DataFrame(data_jap).drop_duplicates()
data_jap.lat, data_jap.lon = data_jap.lat.astype(float), data_jap.lon.astype(float)

'''Merging fluxes with metadata'''
flux_merged = df[df['site'].str.startswith('FLX')].merge(
    flux[['SITE_ID', 'LAT', 'LON', 'IGBP']],
    left_on='site',
    right_on='SITE_ID',
    how='left'
)

ameri_merged = df[df['site'].str.startswith('AMF')].merge(
    ameri[['Site ID', 'Latitude (degrees)', 'Longitude (degrees)', 'Vegetation Abbreviation (IGBP)']],
    left_on='site',
    right_on='Site ID',
    how='left'
)

icos_merged = df[df['site'].str.startswith('ICOS')].merge(
    icos[['site', 'LAT', 'LON', 'IGBP']],
    left_on='site',
    right_on='site',
    how='left'
)

jap_merged = df[df['site'].str.startswith('JPX')].merge(
    data_jap,
    left_on='site',
    right_on='site',
    how='left'
)

df.loc[df['site'].str.startswith('FLX'), ['lat', 'lon', 'IGBP']] = flux_merged[['LAT', 'LON', 'IGBP']].values
df.loc[df['site'].str.startswith('AMF'), ['lat', 'lon', 'IGBP']] = ameri_merged[['Latitude (degrees)', 'Longitude (degrees)', 'Vegetation Abbreviation (IGBP)']].values
df.loc[df['site'].str.startswith('ICOS'), ['lat', 'lon', 'IGBP']] = icos_merged[['LAT', 'LON', 'IGBP']].values
df.loc[df['site'].str.startswith('JPX'), ['lat', 'lon', 'IGBP']] = jap_merged[['lat', 'lon', 'IGBP']].values
df.dropna(inplace=True)

df.to_parquet(f'../data/target_fluxes.parquet', index=None)
print(f"Targets are stacked!")