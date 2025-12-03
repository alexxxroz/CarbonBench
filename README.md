# CarbonBench

1. [KDD](https://kdd2026.kdd.org/datasets-and-benchmarks-track-call-for-papers/): Feb 1, 2026 (abstracts)/ Feb 8 (papers). 
2. 

# Key Ideas

The representation learning + zero/few-shot angle:
- Most carbon flux work assumes access to labeled flux towers everywhere (unrealistic)
- Pre-trained representations that transfer to new locations/ecosystems are understudied
- This directly addresses the "can we actually deploy this globally" question

"The standardized pipeline argument is also solid - if everyone's preprocessing differently, we're not actually comparing models, we're comparing preprocessing choices. That's a real methodological gap."

constrained stratified train-test split

FEATURE_SETS = {
    'minimal': [  # ~8 vars, good baseline
        'temperature_2m', 'total_precipitation_sum',
        'surface_solar_radiation_downwards_sum',
        'volumetric_soil_water_layer_1', ...
    ],
    'standard': [  # ~15-20 vars, recommended
        ...
    ],
    'full': [  # Everything, for exploration
        ...
    ]
}