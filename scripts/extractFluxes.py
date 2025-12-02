'''
    Run only if you want to reproduce the whole benchmark from scratch!
    The script is supposed to process raw site-level data provided as csv files (initially zipped),
    extract targets and stack them and save as a parquet file.
'''

import pandas as pd
import zipfile
from os import listdir
from os.path import join
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--unzip', action='store_true')
args = parser.parse_args()

unzip = args.unzip
config_fname = '../config.yaml'

with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)

'''Path to raw archives'''
main_path = config['flux_path']['main_path']
flux_path = config['flux_path']['flux_path']
ameri_path = config['flux_path']['ameri_path']
icos_path = config['flux_path']['icos_path']
jap_path = config['flux_path']['jap_path']

ameri_files, flux_files, icos_files, jap_files = listdir(ameri_path), listdir(flux_path), listdir(icos_path), listdir(jap_path)

'''Unzip raw files'''
if unzip:
    print('Unzipping...')
    for f in ameri_files:
        if '.zip' in f:
            i=0
            with zipfile.ZipFile(join(ameri_path, f), 'r') as zip_ref:
                new_name = '_'.join(f.split('_')[:2])
                zip_ref.extractall(join(ameri_path, new_name))
            i+=1
    print(f"{i+1} Ameri archives unzipped")
    for f in flux_files:
        if '.zip' in f:
            i=0
            with zipfile.ZipFile(join(flux_path, f), 'r') as zip_ref:
                new_name = '_'.join(f.split('_')[:2])
                zip_ref.extractall(join(flux_path, new_name))
            i+=1
    print(f"{i+1} FLUXNET2015 archives unzipped")
    for f in icos_files:
        if '.zip' in f and '_ARCHIVE_L2' in f:
            i=0
            with zipfile.ZipFile(join(icos_path, f), 'r') as zip_ref:
                new_name = '_'.join(f.split('_')[:2])
                zip_ref.extractall(join(icos_path, new_name))
            i+=1
    print(f"{i+1} ICOS archives unzipped")

flux_meta = 'FLX_AA-Flx'
cols = ['TIMESTAMP', 'GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50', 'NEE_VUT_USTAR50', 'NEE_VUT_USTAR50_QC']
QC = cols[-1]

flux_dir = [x for x in listdir(flux_path) if (x.split('.')[-1] != 'zip')&(x != flux_meta)]
ameri_dir = [x for x in listdir(ameri_path) if (x.split('.')[-1] != 'zip')]
icos_dir = [x for x in listdir(icos_path) if (len(x.split('.')) == 1)]
jap_dir = [x for x in listdir(jap_path) if (x.split('.')[-1] != 'zip')]

'''Extracting FLUXNET time series'''
print('Processing FLUXNET...')
data = []
for folder in flux_dir:
    try:
        csv = [x for x in listdir(join(flux_path, folder)) if (x.split('_')[3]=='FULLSET') & (x.split('_')[4]=='DD')][0]
        df = pd.read_csv(join(flux_path, folder, csv))[cols]
        df['site'] = ['_'.join(csv.split('_')[:2])]*len(df)
        data.append(df)
    except Exception as e:
        print(e)
        pass
data = pd.concat(data, axis=0)
data = data.reset_index(drop=True)
data.to_parquet(f'{main_path}/all_fluxnet.parquet', index=None)
print(f"FLUXNET is processed: extracted {data.site.nunique()} sites")

'''Extracting AmeriFlux time series'''
print('Processing AmeriFlux...')
data_ameri = []
for folder in ameri_dir:
    try:
        csv = [x for x in listdir(join(ameri_path, folder)) if (x.split('_')[3]=='FULLSET') & (x.split('_')[4]=='DD')][0]
        df = pd.read_csv(join(ameri_path, folder, csv))[cols]
        df['site'] = ['_'.join(csv.split('_')[:2])] * len(df)
        data_ameri.append(df)
    except:
        pass
data_ameri = pd.concat(data_ameri, axis=0)
data_ameri = data_ameri.reset_index(drop=True)
data_ameri.to_parquet(f'{main_path}/all_ameri.parquet', index=None)
print(f"AmeriFlux is processed: extracted {data_ameri.site.nunique()} sites")

'''Extracting ICOS time series'''
print('Processing ICOS...')
data_icos, igbps = [], {}
for folder in icos_dir:
    try:
        csv = f'{folder}_FLUXNET_DD_L2.csv'
        df = pd.read_csv(join(icos_path, folder, csv))[cols]
        df['site'] = ['_'.join(csv.split('_')[:2])]*len(df)
        data_icos.append(df)

        site_info = f'{folder}_SITEINFO_L2.csv'
        df = pd.read_csv(join(icos_path, folder, site_info))
        IGBP = df[df.VARIABLE=='IGBP']['DATAVALUE'].item()
        igbps[folder] = [IGBP]
    except Exception as e:
        pass
data_icos = pd.concat(data_icos, axis=0)
data_icos = data_icos.reset_index(drop=True)
IGBP = pd.DataFrame(igbps).T.reset_index(drop=False).rename(columns={0: 'IGBP', 'index': 'site'})
data_icos.to_parquet(f'{main_path}/all_icos.parquet', index=None)
IGBP.to_csv(f'{main_path}/icos_IGBPs.csv', index=None)
print(f"ICOS is processed: extracted {data_icos.site.nunique()} sites")

'''Extracting JapanFlux time series'''
print('Processing JapanFlux...')
data_jap = []
for folder in jap_dir:
    try:
        csv = [x for x in listdir(join(jap_path, folder, folder.split('_')[0], 'DATA','COREVARS')) if (x.split('_')[4]=='DD')][0]
        df = pd.read_csv(join(jap_path, folder, folder.split('_')[0], 'DATA','COREVARS', csv))
        df.columns = df.columns.str.upper()
        df = df[cols]
        df = df.replace(-9999, None).dropna(subset='NEE_VUT_USTAR50_QC')
        df['NEE_VUT_USTAR50_QC'] = (3 - df['NEE_VUT_USTAR50_QC']) / 3 # 0 - best, 3 - worst -> convert to continuous binary
        df['site'] = 'JPX_' + csv.split('_')[1]
        data_jap.append(df)
    except Exception as e:
        print(e)
        pass
if len(data_jap) > 0:
    data_jap = pd.concat(data_jap, axis=0).drop_duplicates()
    data_jap = data_jap.reset_index(drop=True)
    data_jap.to_parquet(f'{main_path}/all_jap.parquet', index=None)
    print(f"JapanFlux is processed: extracted {data_jap.site.nunique()} sites")
else:
    print('No JapanFlux data exist with these vars')

'''Stacking all'''
if type(data_jap)==pd.DataFrame:
    all_data = pd.concat([data, data_ameri, data_icos, data_jap], axis=0).reset_index(drop=True)
else:
    all_data = pd.concat([data, data_ameri, data_icos], axis=0).reset_index(drop=True)    
all_data.to_parquet(f'{main_path}/all_fluxes.parquet', index=None)
print(f'Done! Saved {all_data.site.nunique()} sites.')