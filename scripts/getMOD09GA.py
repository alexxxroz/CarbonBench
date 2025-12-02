'''
    Run only if you want to reproduce the whole benchmark from scratch!
    To successfully run the script you need login into your Google Earth Engine (GEE) account beforehand and specify a Google Cloud project in ../config.yaml.
    The default values in config.yaml will lead to code failure.
    The script reads the parquet file with target fluxes and derives MOD09GA features for every site.
    In particular, 2x2 km squares is created for every location and centered in the site coordinates. Then MODIS pixels overlaping with the created buffer are averaged and saved.
'''


import os
import glob
import yaml
import shutil

import pandas as pd
import numpy as np

import ee

def getCollection(lat, lon, start_year, end_year):
    roi = ee.Geometry.Point([lon, lat])
    data = ee.ImageCollection("MODIS/061/MOD09GA") \
                .filterBounds(roi) \
                .filterDate(f'{start_year}-01-01', f'{end_year}-01-01') \
                .select(bands)
    return data, roi

def mask(image):
    qa = image.select(bands[-1])
    cloud  = qa.bitwiseAnd(1 << 0).eq(1)
    shadow = qa.bitwiseAnd(1 << 1).eq(1)
    cirrus = qa.bitwiseAnd(1 << 2).eq(1)
    return cloud.Or(shadow).Or(cirrus)      \
                .rename('clouds')          \
                .toInt16()

def make_reducer(buffered_roi):
    def _fn(image):
        cloud_mask = mask(image)  
        bands_no_QA = image.bandNames().removeAll(['state_1km'])
        img = image.select(bands_no_QA).addBands(cloud_mask)
        vals = img.reduceRegion(
            ee.Reducer.mean(),
            buffered_roi,
            500,
            bestEffort=True,
            maxPixels=1e13
        )
        return (ee.Feature(None, vals).set('date', image.date().format('YYYY-MM-dd')))
    return _fn

config_fname = '../config.yaml'
with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)

ee.Authenticate()
ee.Initialize(project=config['gee_project'])

fluxes = pd.read_parquet('../data/target_fluxes.parquet')
fluxes['TIMESTAMP'] = pd.to_datetime(fluxes.TIMESTAMP, format='%Y%m%d')

surf_refl = [f'sur_refl_b0{i}' for i in range(1,8)]
bands = surf_refl + ['SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth', 'state_1km']

if os.path.exists('../data/MOD09GA'):
    sites = [x.split('.')[0] for x in sorted(os.listdir('../data/MOD09GA'))]
else:
    sites = []

os.makedirs('../data/MOD09GA/', exist_ok=True)
for site, group in fluxes.groupby(['site']):
    if site[0] not in sites:
        lat, lon = group.lat.unique()[0], group.lon.unique()[0]
        d = {x: [] for x in ['site', 'date'] + bands[:-1] + ['clouds']}
        for start_year, end_year in zip([2000, 2008, 2016], [2008, 2016, 2024]): #GEE can't process seq longer than 5k 
            collection, roi = getCollection(lat, lon, start_year, end_year)
            buffered = roi.buffer(1000).bounds() # 2km by 2km
            reducer = make_reducer(buffered)
            
            feature_collection = collection.map(reducer)

            for feature in feature_collection.getInfo( )['features']:
                image = feature['properties']
                if all(k in feature['properties'].keys() for k in list(d.keys())[1:]):
                    for band in surf_refl:
                        image[band] *= 0.0001
                    for band in ['SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth']:
                        image[band] *= 0.01

                    d['site'].append(site[0])
                    for key in image.keys():
                        d[key].append(image[key])
        n_samples = len(np.where(np.array(d['site'])==site[0])[0])
        site_df = pd.DataFrame(d).drop_duplicates()
        site_df.to_parquet(f'../data/MOD09GA/{site[0]}.parquet', index=False)
        print(f'{site[0]} saved with {n_samples} samples')

dfs = [pd.read_parquet(f) for f in glob.glob('../data/MOD09GA/*.parquet')]
pd.concat(dfs).to_parquet('../data/MOD09GA.parquet', index=False)

if len(dfs)==fluxes.site.nunique():
    shutil.rmtree('../data/MOD09GA/')