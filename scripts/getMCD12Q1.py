'''
    Run only if you want to reproduce the whole benchmark from scratch!
    To successfully run the script you need login into your Google Earth Engine (GEE) account beforehand and specify a Google Cloud project in ../config.yaml.
    The default values in config.yaml will lead to code failure.
    The script reads the parquet file with target fluxes and derives MCD12Q1 features for every site.
    In particular, 2x2 km squares is created for every location and centered in the site coordinates. Then MODIS pixels overlaping with the created buffer are processed using majority
    voting (the feature is categorical) and saved.
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
    data = ee.ImageCollection("MODIS/061/MCD12Q1") \
                .filterBounds(roi) \
                .filterDate(f'{start_year}-01-01', f'{end_year}-01-01') \
                .select(bands)
    return data, roi

def make_reducer(buffered_roi):
    def _fn(image):
        vals = image.reduceRegion(
            ee.Reducer.mode(), # majority class 
            buffered_roi,
            500,
            bestEffort=True,
            maxPixels=1e13
        )
        
        return ee.Feature(None, vals).set(
            'date', image.date().format('YYYY-MM-dd')
        )
    return _fn

config_fname = '../config.yaml'
with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)

ee.Authenticate()
ee.Initialize(project=config['gee_project'])

fluxes = pd.read_parquet('../data/target_fluxes.parquet')
fluxes['TIMESTAMP'] = pd.to_datetime(fluxes.TIMESTAMP, format='%Y%m%d')

bands = ['LC_Type1']
landcover_to_igbp = {
    1: 'ENF',  # Evergreen Needleleaf Forest
    2: 'EBF',  # Evergreen Broadleaf Forest
    3: 'DNF',  # Deciduous Needleleaf Forest
    4: 'DBF',  # Deciduous Broadleaf Forest
    5: 'MF',   # Mixed Forest
    6: 'CSH',  # Closed Shrublands
    7: 'OSH',  # Open Shrublands
    8: 'WSA',  # Woody Savanna
    9: 'SAV',  # Savanna
    10: 'GRA', # Grasslands
    11: 'WET', # Wetlands
    12: 'CRO', # Croplands
    13: 'URB', # Urban
    14: 'CVM', # Cropland/Natural Vegetation Mosaic
    15: 'SNO', # Snow and Ice
    16: 'BSV', # Barren or Sparse Vegetation
    17: 'WAT'  # Water Bodies
}

if os.path.exists('../data/MCD12Q1.csv'):
    df = pd.read_csv('../data/MCD12Q1.csv')
    sites = np.unique(df.to_dict(orient='list')['site'])
else:
    sites = []

os.makedirs('../data/MCD12Q1/', exist_ok=True)
for site, group in fluxes.groupby(['site']):
    if site not in sites:
        lon, lat = group.lon.unique()[0], group.lat.unique()[0]

        start_year, end_year = 2000, 2024
        collection, roi = getCollection(lat, lon, start_year, end_year)
        buffered = roi.buffer(1000).bounds() # 2km by 2km
        reducer = make_reducer(roi)

        feature_collection = collection.map(reducer)
        
        d = {x: [] for x in ['site', 'date'] + bands}
        for feature in feature_collection.getInfo()['features']:
            image = feature['properties']
            if all(k in feature['properties'].keys() for k in list(d.keys())[1:]):

                d['site'].append(site[0])
                for key in image.keys():
                    if key=='LC_Type1':
                        d[key].append(landcover_to_igbp[image[key]])
                    else:
                        d[key].append(image[key])
        n_samples = len(np.where(np.array(d['site'])==site[0])[0])
        site_df = pd.DataFrame(d).drop_duplicates()
        site_df.date = pd.to_datetime(site_df.date)
        site_df = site_df.set_index('date').resample('1D').ffill().reset_index()

        site_df.to_csv('../data/MCD12Q1.csv', mode='a', 
                       header=not os.path.exists('../data/MCD12Q1.csv'), 
                       index=False)
        print(f'{site[0]} saved with {n_samples} samples')