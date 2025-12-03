'''
    Run only if you want to reproduce the whole benchmark from scratch!
    To successfully run the script you need to have fluxes pre-processed.
    The script extract koppen climate classes for every site.
'''


import json
import yaml
import numpy as np
import pandas as pd
import xarray as xr

config_fname = '../config.yaml'
with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)
path = config['koppen_path']

with open(f'{path}/koppen_short_codes.json', 'r') as file:
    short_codes = json.load(file)
with open(f'{path}/koppen_classes.json', 'r') as file:
    classes = json.load(file)
    classes = {int(k):str(v) for k,v in classes.items()}

y = pd.read_parquet('../data/target_fluxes.parquet')    
    
koppen = xr.open_dataset(f'{path}/koppen_geiger_0p00833333.nc')
res = {}
for site in y.site.unique():
    lat, lon = y[y.site==site].lat.values[0], y[y.site==site].lon.values[0]
    k = koppen.sel(lat=lat, lon=lon, method='nearest').kg_class.item()
    if k==0:
        # if koppen is Null, slice a chunck of Nearest Neighbors, exclude 0s (null) and pick the majority class
        NN = koppen.sel(lat=slice(lat+0.1, lat-0.1),lon=slice(lon-0.1, lon+0.1)).kg_class.to_numpy()
        NN = NN[NN!=0]
        vals, counts = np.unique(NN, return_counts=True)
        k = vals[np.argmax(counts)] 
    res[site] = short_codes[classes[k]]

with open("../data/koppen_sites.json", "w") as outfile:
        json.dump(res, outfile)
print("Koppen data successfully written to ../data/koppen_sites.json")