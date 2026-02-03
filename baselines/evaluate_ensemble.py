# Evaluate ensemble models by averaging predictions across 10 seeds; load all 10 seed models, average their predictions, then compute metrics

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import r2_score, root_mean_squared_error

import carbonbench


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ensemble models on test set')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'ctlstm', 'gru', 'ctgru', 'transformer', 'patch_transformer'],
                        help='Model architecture')
    parser.add_argument('--split_type', type=str, required=True,
                        choices=['IGBP', 'Koppen'],
                        help='Split type')
    parser.add_argument('--task_type', type=str, required=True,
                        choices=['zero-shot', 'few-shot'])
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file with best hyperparameters')
    parser.add_argument('--model_dir', type=str, default='./outputs',
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model(model_name, **kwargs):
    model_map = {
        'lstm': carbonbench.lstm,
        'ctlstm': carbonbench.ctlstm,
        'gru': carbonbench.gru,
        'ctgru': carbonbench.ctgru,
        'transformer': carbonbench.transformer,
        'patch_transformer': carbonbench.patch_transformer,
    }
    return model_map[model_name](**kwargs)


def mape(mean_flux, true, pred):
    mape_val = np.abs(pred - true) / (np.abs(mean_flux) + 1e-9)
    return np.mean(mape_val)


def get_predictions(model, loader, model_name, device):
    """Get predictions from a single model for all samples in loader."""
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for x, x_static, y, _, _, _ in loader:
            x = x.to(device)
            x_static = x_static.to(device)
            
            if model_name in ['lstm', 'gru']:
                pred = model(x)
            else:
                pred = model(x, x_static)
            
            # take last timestep prediction
            all_preds.append(pred[:, -1, :].cpu().numpy())
            
            # FIXED: take last timestep for y as well (y shape: batch, stride, n_targets)
            y_cpu = y[:, -1, :].cpu().numpy()
            all_true.append(y_cpu)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    
    return all_preds, all_true


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {args.model}, Split: {args.split_type}")
    
    config = load_config(args.config)
    best_params = config['best_params']
    print(f"Best params: {best_params}")
    
    # Define all 10 seeds
    seeds = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    # Check which seed models exist
    available_seeds = []
    for seed in seeds:
        model_subdir = f"{args.model}_seed{seed}_{args.split_type}"
        model_path = os.path.join(args.model_dir, model_subdir, 'model.pt')
        if os.path.exists(model_path):
            available_seeds.append(seed)
        else:
            print(f"Warning: Model not found for seed {seed}: {model_path}")
    
    if len(available_seeds) == 0:
        print("No models found, exiting.")
        sys.exit(1)
    
    print(f"Found {len(available_seeds)} seed models: {available_seeds}")
    
    print(f"\nLoading data...")
    targets = ['GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50', 'NEE_VUT_USTAR50']
    include_qc = True
    test_QC_threshold = 1
    window_size = 30
    stride = 15
    
    y = carbonbench.load_targets(targets, include_qc)
    if args.task_type=='zero-shot':
        y_train, y_test = carbonbench.split_targets(
            y, task_type=args.task_type, split_type=args.split_type,
            verbose=False, plot=False
        )
    elif args.task_type=='few-shot':
        y_train, y_test, y_finetune = carbonbench.split_targets(
            y, task_type=args.task_type, split_type=args.split_type, 
            verbose=False, plot=False
        )
    
    modis = carbonbench.load_modis()
    era = carbonbench.load_era('minimal')
    
    if args.task_type=='zero-shot':
        train_cv, val_cv, test, x_scaler, y_scaler = carbonbench.join_features(
            y_train, y_test, modis, era, val_ratio=0.2, scale=True
        )
        
        train_hist = carbonbench.historical_cache(train_cv, era, modis, x_scaler, window_size)
        train_dataset = carbonbench.SlidingWindowDataset(
            train_hist, targets, include_qc,
            window_size=window_size, stride=stride,
            cat_features=['IGBP', 'Koppen', 'Koppen_short']
        )
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.task_type=='few-shot':
        train, val, finetune, test, x_scaler, y_scaler = carbonbench.join_features(
            y_train, y_test, modis, era, val_ratio=0.2, scale=True
        )
        
        finetune_hist = carbonbench.historical_cache(finetune, era, modis, x_scaler, window_size)
        finetune_dataset = carbonbench.SlidingWindowDataset(finetune_hist, targets, include_qc, window_size=window_size, 
                                                            stride=1, cat_features=['IGBP', 'Koppen', 'Koppen_short'])
        loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    encoders = loader.encoders  
    test_hist = carbonbench.historical_cache(test, era, modis, x_scaler, window_size)
    test_dataset = carbonbench.SlidingWindowDataset(
        test_hist, targets, include_qc,
        window_size=window_size, stride=1, QC_threshold=test_QC_threshold,
        encoders=encoders,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )
    
    x_sample, x_static_sample, _, _, _, _ = next(iter(loader))
    input_dynamic_channels = x_sample.shape[2]
    input_static_channels = x_static_sample.shape[2]
    output_channels = len(targets)
    
    # Build model kwargs
    model_kwargs = {
        'input_dynamic_channels': input_dynamic_channels,
        'hidden_dim': best_params['hidden_dim'],
        'output_channels': output_channels,
        'dropout': best_params['dropout'],
    }
    
    if args.model not in ['lstm', 'gru']:
        model_kwargs['input_static_channels'] = input_static_channels
    
    if 'transformer' in args.model:
        model_kwargs['nhead'] = best_params['nhead']
        model_kwargs['num_layers'] = best_params['num_layers']
        model_kwargs['seq_len'] = window_size
        if 'patch' in args.model:
            model_kwargs['pred_len'] = stride
            model_kwargs['patch_len'] = 10
            model_kwargs['stride'] = 5
    else:
        model_kwargs['layers'] = best_params['layers']
    
    results = {target: {'site': [], 'IGBP': [], 'Koppen': [], 
                        'R2': [], 'RMSE': [], 'MAPE': [],
                        'pred_std': []} for target in targets}
    
    for site in test_dataset.get_sites():
        site_indices = test_dataset.get_site_indices(site)
        site_subset = Subset(test_dataset, site_indices)
        site_loader = DataLoader(site_subset, batch_size=args.batch_size, shuffle=False)
        
        # Collect predictions from all seed models
        all_seed_preds = []
        true_values = None
        
        for seed in available_seeds:
            model_subdir = f"{args.model}_seed{seed}_{args.split_type}"
            model_path = os.path.join(args.model_dir, model_subdir, 'model.pt')
            
            model = get_model(args.model, **model_kwargs)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            
            preds, true = get_predictions(model, site_loader, args.model, device)
            all_seed_preds.append(preds)
            
            if true_values is None:
                true_values = true
        
        # Stack predictions from all seeds: (n_seeds, n_samples, n_targets)
        all_seed_preds = np.stack(all_seed_preds, axis=0)
        
        # Average predictions across seeds
        ensemble_preds = np.mean(all_seed_preds, axis=0)
        
        # Compute prediction std across seeds (uncertainty measure)
        pred_std = np.std(all_seed_preds, axis=0)
        
        # inverse transform to original scale
        ensemble_preds = y_scaler.inverse_transform(ensemble_preds)
        true_values = y_scaler.inverse_transform(true_values)
        pred_std_orig = pred_std * y_scaler.scale_
        
        # compute metrics for each target
        for idx, target in enumerate(targets):
            y_true = true_values[:, idx]
            y_pred = ensemble_preds[:, idx]
            
            r2 = np.clip(r2_score(y_true, y_pred), 0, 1)
            rmse = root_mean_squared_error(y_true, y_pred)
            mape_val = mape(np.mean(y_true), y_true, y_pred)
            mean_pred_std = np.mean(pred_std_orig[:, idx])
            
            results[target]['site'].append(site)
            results[target]['IGBP'].append(test_cv[test_cv.site == site].IGBP.values[0])
            results[target]['Koppen'].append(test_cv[test_cv.site == site].Koppen.values[0])
            results[target]['R2'].append(r2)
            results[target]['RMSE'].append(rmse)
            results[target]['MAPE'].append(mape_val)
            results[target]['pred_std'].append(mean_pred_std)
    
    print("\nEnsemble Results (averaged predictions from {} seeds):".format(len(available_seeds)))
    print("\t\t\t\tR2\tRMSE\tMAPE\tPred_STD")
    for target in targets:
        r2_mean = np.mean(results[target]['R2'])
        rmse_mean = np.mean(results[target]['RMSE'])
        mape_mean = np.mean(results[target]['MAPE'])
        std_mean = np.mean(results[target]['pred_std'])
        print(f"{target}:\t{r2_mean:.2f}\t{rmse_mean:.2f}\t{mape_mean:.2f}\t{std_mean:.2f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    for target in targets:
        df = pd.DataFrame(results[target])
        df['target'] = target
        df['model'] = args.model
        df['split_type'] = args.split_type
        df['n_seeds'] = len(available_seeds)
        all_results.append(df)
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    # output no longer includes seed (we use all seeds)
    output_file = os.path.join(args.output_dir, f"{args.model}_{args.split_type}_ensemble_{args.task_type}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()