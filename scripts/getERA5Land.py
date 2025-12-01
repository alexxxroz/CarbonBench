import os
import yaml
import glob
import shutil

import pandas as pd
import numpy as np

import ee

def getCollection(lat, lon, start_year, end_year):
    roi = ee.Geometry.Point([lon, lat])
    data = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterBounds(roi) \
                .filterDate(f'{start_year}-01-01', f'{end_year}-01-01') \
                .select(bands)
    return data, roi

def make_reducer(buffered_roi):
    def _fn(img):
        # bilinear interpolation is needed since ~10 sites are masked out in ERA5 Land since proximity to water; 
        # interpolations allows to preserve these sites
        vals = img.resample('bilinear')reduceRegion( 
            ee.Reducer.mean(),
            buffered_roi,
            scale=11132, 
            bestEffort=True,
            maxPixels=1e13
        )
        
        return ee.Feature(None, vals).set(
            'date', img.date().format('YYYY-MM-dd')
        )
    return _fn

config_fname = '../config.yaml'
with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)

ee.Authenticate()
ee.Initialize(project=config['gee_project'])

fluxes = pd.read_parquet('../../data/fluxes/target_fluxes.parquet')
fluxes['TIMESTAMP'] = pd.to_datetime(fluxes.TIMESTAMP, format='%Y%m%d')

# Extract all band names from ERA5 Land daily agg (150 features)
data = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterDate(f'2001-01-01', f'2001-01-02') 
bands = []
for band in data.getInfo()['features'][0]['bands']:
    bands.append(band['id'])

if os.path.exists('../data/ERA5.parquet'):
    df = pd.read_parquet('../data/ERA5.parquet')
    d = df.to_dict(orient='list')
    sites = np.unique(df.to_dict(orient='list')['site'])
else:
    sites = []

os.makedirs('../data/ERA5/', exist_ok=True)
for site, group in fluxes.groupby(['site']):
    if site not in sites:
        lat, lon = group.lat.unique()[0], group.lon.unique()[0]
        
        d = {x: [] for x in ['site', 'date'] + bands}
        for start_year, end_year in zip([2000, 2008, 2016], [2008, 2016, 2024]):
            collection, roi = getCollection(lat, lon, start_year, end_year)
            buffered = roi.buffer(1000).bounds() # 2km by 2km
            reducer = make_reducer(buffered)
            
            feature_collection = collection.map(reducer)

            for feature in feature_collection.getInfo()['features']:
                image = feature['properties']
                if all(k in image for k in list(d.keys())[1:]):
                    d['site'].append(site[0])
                    for key in image:
                        d[key].append(image[key])
            
        n_samples = len(d['site']) 
        site_df = pd.DataFrame(d).drop_duplicates()
        site_df.to_parquet(f'../data/ERA5/{site[0]}.parquet', index=None)
        print(f'{site[0]} is saved with {n_samples} samples')
        
dfs = [pd.read_parquet(f) for f in glob.glob('../data/ERA5/*.parquet')]
pd.concat(dfs).to_parquet('../data/ERA5.parquet', index=False)

if len(dfs)==fluxes.site.nunique():
    shutil.rmtree('../data/ERA5/')