# Extract best hyperparameters from grid search results and save as YAML configs

import os
import yaml
import pandas as pd
import argparse


# These columns are metrics, not hyperparameters
METRIC_COLUMNS = [
    'cv_r2_mean', 
    'cv_r2_std', 
    'cv_rmse_mean', 
    'cv_rmse_std', 
    'cv_val_loss_mean', 
    'fold_r2_scores'
]


def extract_best_params(input_dir, output_dir):
    """
    Read gridsearch_results.csv from each subdirectory in input_dir,
    find the best hyperparameters by cv_r2_mean,
    and save them as YAML files in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    
    models_to_process = ['lstm', 'gru', 'ctlstm', 'ctgru', 'transformer']
    
    subdirs = [d for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d))]
    
    subdirs = [d for d in subdirs if any(d.startswith(m) for m in models_to_process)]
    
    if not subdirs:
        print(f"No subdirectories found in {input_dir}")
        return
    
    print(f"Found {len(subdirs)} result directories")
    
    
#     # Find all subdirectories with results
#     subdirs = [d for d in os.listdir(input_dir) 
#                if os.path.isdir(os.path.join(input_dir, d))]
    
#     if not subdirs:
#         print(f"No subdirectories found in {input_dir}")
#         return
    
#     print(f"Found {len(subdirs)} result directories")
    
    for subdir in sorted(subdirs):
        csv_path = os.path.join(input_dir, subdir, 'gridsearch_results.csv')
        
        if not os.path.exists(csv_path):
            print(f"Skipping {subdir}: no gridsearch_results.csv found")
            continue
        
        # Read results
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"Skipping {subdir}: empty CSV file")
            continue
        
        if 'cv_r2_mean' not in df.columns:
            print(f"Skipping {subdir}: cv_r2_mean column not found")
            continue
        
        # Find best row by cv_r2_mean
        best_idx = df['cv_r2_mean'].idxmax()
        best_row = df.loc[best_idx]
        
        # Extract hyperparameter columns only
        hyperparam_cols = [col for col in df.columns if col not in METRIC_COLUMNS]
        best_params = {col: best_row[col] for col in hyperparam_cols}
        
        # Convert numpy types to Python native types for YAML
        for key, value in best_params.items():
            if hasattr(value, 'item'):
                best_params[key] = value.item()
        
        # Parse model and split_type from directory name
        parts = subdir.rsplit('_', 1)
        if len(parts) == 2:
            model_name, split_type = parts
        else:
            model_name = subdir
            split_type = 'unknown'
        
        # Build config dict
        config = {
            'model': model_name,
            'split_type': split_type,
            'best_params': best_params,
            'cv_r2_mean': float(best_row['cv_r2_mean']),
            'cv_r2_std': float(best_row['cv_r2_std']),
            'cv_rmse_mean': float(best_row['cv_rmse_mean']),
        }
        
        # Save to YAML
        yaml_path = os.path.join(output_dir, f'{subdir}.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Saved {yaml_path}")
        print(f"  Best R2: {config['cv_r2_mean']:.4f} +/- {config['cv_r2_std']:.4f}")
        print(f"  Params: {best_params}")
        print()
    
    print(f"Done. Config files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract best hyperparameters from grid search results')
    parser.add_argument('--input_dir', type=str, default='./gridsearch_outputs',
                        help='Directory containing grid search results')
    parser.add_argument('--output_dir', type=str, default='./configs',
                        help='Directory to save YAML config files')
    args = parser.parse_args()
    
    extract_best_params(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()