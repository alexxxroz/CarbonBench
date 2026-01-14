# Grid search for hyperparameter tuning using 5-fold CV

import os
import sys
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

import carbonbench

from utils import train_tamrl

def parse_args():
    parser = argparse.ArgumentParser(description='CarbonBench Hyperparameter Grid Search')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'ctlstm', 'gru', 'ctgru', 'transformer', 'patch_transformer', 'tam-rl'],
                        help='Model architecture')
    parser.add_argument('--split_type', type=str, required=True,
                        choices=['IGBP', 'Koppen'],
                        help='Train-test split stratification type')
    parser.add_argument('--output_dir', type=str, default='./gridsearch_outputs',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=27,
                        help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (stop if no improvement for this many epochs)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, **kwargs):
    model_map = {
        'lstm': carbonbench.lstm,
        'ctlstm': carbonbench.ctlstm,
        'gru': carbonbench.gru,
        'ctgru': carbonbench.ctgru,
        'transformer': carbonbench.transformer,
        'patch_transformer': carbonbench.patch_transformer,
        'tam-rl': carbonbench.lstm # for TAM-RL use lstm first to pre-train the decoder, and then switch to AE and TAMLSTM
    }
    return model_map[model_name](**kwargs)

def get_param_grid(model_name):
    # grid for actual search
    if model_name in ['lstm', 'gru', 'ctlstm', 'ctgru']:
        return {
            'hidden_dim': [64, 128, 256],
            'dropout': [0.1, 0.2, 0.3],
            'lr': [5e-4, 1e-3, 2e-3],
            'layers': [1, 2],
        }
    elif model_name=='tam-rl':
        return {
            'hidden_dim': [64, 128, 256],
            'latent_dim': [32, 64],
            'dropout': [0.1, 0.2, 0.3],
            'lr': [5e-4, 1e-3, 2e-3],
            'layers': [1, 2],
        }
    else:  # transformer, patch_transformer
        return {
            'hidden_dim': [64, 128, 256],
            'dropout': [0.1, 0.2, 0.3],
            'lr': [1e-4, 5e-4, 1e-3],
            'nhead': [2, 4, 8],
            'num_layers': [2, 4],
        }

def train_one_fold(args, fold, hyperparams, y_train, modis, era, cv_split, device, patience=10):
    targets = ['GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50', 'NEE_VUT_USTAR50']
    include_qc = True
    test_QC_threshold = 1
    window_size = 30
    stride = 15
    num_workers = 4
    
    # Split by fold
    test_sites = cv_split['folds'][str(fold)]
    y_train_cv = y_train[~y_train.site.isin(test_sites)]
    y_test_cv = y_train[y_train.site.isin(test_sites)]
    
    train_cv, val_cv, test_cv, x_scaler, y_scaler = carbonbench.join_features(
        y_train_cv, y_test_cv, modis, era, val_ratio=0.2, scale=True
    )
    
    # Create data loaders
    train_hist = carbonbench.historical_cache(train_cv, era, modis, x_scaler, window_size)
    train_dataset = carbonbench.SlidingWindowDataset(
        train_hist, targets, include_qc,
        window_size=window_size, stride=stride,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False
    )
    
    val_hist = carbonbench.historical_cache(val_cv, era, modis, x_scaler, window_size)
    val_dataset = carbonbench.SlidingWindowDataset(
        val_hist, targets, include_qc,
        window_size=window_size, stride=stride,
        encoders=train_dataset.encoders,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    test_hist = carbonbench.historical_cache(test_cv, era, modis, x_scaler, window_size)
    test_dataset = carbonbench.SlidingWindowDataset(
        test_hist, targets, include_qc,
        window_size=window_size, stride=1, QC_threshold=test_QC_threshold,
        encoders=train_dataset.encoders,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )
    
    # Get dimensions
    x_sample, x_static_sample, _, _, _, _ = next(iter(train_loader))
    input_dynamic_channels = x_sample.shape[2]
    input_static_channels = x_static_sample.shape[2]
    output_channels = len(targets)
    
    # Build model
    model_kwargs = {
        'input_dynamic_channels': input_dynamic_channels,
        'hidden_dim': hyperparams['hidden_dim'],
        'output_channels': output_channels,
        'dropout': hyperparams['dropout'],
    }
    
    if args.model in ['lstm', 'gru']:
        model_kwargs['layers'] = hyperparams['layers']
    else:
        model_kwargs['input_static_channels'] = input_static_channels
    
    if 'transformer' in args.model:
        model_kwargs['nhead'] = hyperparams['nhead']
        model_kwargs['num_layers'] = hyperparams['num_layers']
        model_kwargs['seq_len'] = window_size
        if 'patch' in args.model:
            model_kwargs['pred_len'] = stride
            model_kwargs['patch_len'] = 10
            model_kwargs['stride'] = 5
    else:
        model_kwargs['layers'] = hyperparams['layers']
    
    if 'tam-rl' in args.model:
        model_kwargs['latent_dim'] = hyperparams['latent_dim']
        
    model = get_model(args.model, **model_kwargs).to(device)
    
    # Loss and optimizer
    IGBP = train_cv['IGBP'].values
    IGBP_weights = compute_class_weight(class_weight="balanced", classes=np.unique(IGBP), y=IGBP)
    IGBP_weights = {str(k): float(IGBP_weights[i]) for i, k in enumerate(np.unique(IGBP))}
    
    Koppen = train_cv['Koppen'].values
    Koppen_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Koppen), y=Koppen)
    Koppen_weights = {str(k): float(Koppen_weights[i]) for i, k in enumerate(np.unique(Koppen))}
    
    criterion = carbonbench.CustomLoss(IGBP_weights, Koppen_weights, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams['lr'])
    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_count = 0
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for x, x_static, y_batch, qc, igbp_w, koppen_w in train_loader:
            x = x.to(device)
            x_static = x_static.to(device)
            y_batch = y_batch.to(device)
            qc = qc.to(device)
            igbp_w = igbp_w.to(device)
            koppen_w = koppen_w.to(device)
            
            optimizer.zero_grad()
            
            if model.__class__.__name__ in ['lstm', 'gru']:
                pred = model(x)
            else:
                pred = model(x, x_static)
            
            loss = criterion(pred[:, -stride:, :], y_batch[:, -stride:, :3], qc, igbp_w, koppen_w)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if epoch % 5==0:
            model.eval()
            val_preds = []
            val_true = []

            with torch.no_grad():
                for x, x_static, y_batch, _, _, _ in val_loader:
                    x = x.to(device)
                    x_static = x_static.to(device)
                    y_batch = y_batch.to(device)

                    if model.__class__.__name__ in ['lstm', 'gru']:
                        pred = model(x)
                    else:
                        pred = model(x, x_static)

                    val_preds.append(pred.cpu())
                    val_true.append(y_batch.cpu())

            val_preds = torch.cat(val_preds)
            val_true = torch.cat(val_true)
            val_loss = criterion(
                val_preds[:, -stride:, :].to(device),
                val_true[:, -stride:, :3].to(device)
            ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_model = model.state_dict()
                no_improve_count = 0  
            else:
                no_improve_count += 5 # since validated every 5 epochs

            if no_improve_count >= patience:
                break
    
    # Load best model and evaluate
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if args.model=='tam-rl':
        inverse_model = carbonbench.ae_tamrl(input_channels=model_kwargs['input_dynamic_channels']+model_kwargs['input_static_channels'], 
                                             code_dim=model_kwargs['latent_dim'], hidden_dim=model_kwargs['latent_dim'], output_channels=model_kwargs['latent_dim']).to(device)
        forward_model = carbonbench.tamlstm(model_kwargs['input_dynamic_channels'], model_kwargs['latent_dim'], model_kwargs['hidden_dim'], 
                                            model_kwargs['output_channels'], model_kwargs['dropout']).to(device)

        encoder_weights = {k.replace('encoder.', ''): v for k, v in best_model.items() if k.startswith('encoder.')}
        forward_model.encoder.load_state_dict(encoder_weights)

        criterion = carbonbench.CustomLoss(IGBP_weights, Koppen_weights) 
        optimizer = optim.Adam(list(inverse_model.parameters())+list(forward_model.parameters()), lr=1e-3) 
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        
        train_dataset_tamrl = carbonbench.SlidingWindowDatasetTAMRL(train_hist, targets, include_qc, 
                                                                    window_size=window_size, stride=stride, cat_features=['IGBP', 'Koppen', 'Koppen_short'])
        train_loader_tamrl = DataLoader(train_dataset_tamrl, batch_size=args.batch_size, shuffle=True)

        val_dataset_tamrl = carbonbench.SlidingWindowDatasetTAMRL(val_hist, targets, include_qc, window_size=window_size, stride=stride, 
                                                                  encoders=train_dataset.encoders, cat_features=['IGBP', 'Koppen', 'Koppen_short'])
        val_loader_tamrl = DataLoader(val_dataset_tamrl, batch_size=args.batch_size, shuffle=True)

        test_dataset_tamrl = carbonbench.SlidingWindowDatasetTAMRL(test_hist, targets, include_qc, window_size=window_size, QC_threshold=test_QC_threshold, 
                                                                   stride=1, cat_features=['IGBP', 'Koppen', 'Koppen_short'], encoders=train_dataset.encoders)
        test_loader_tamrl = DataLoader(test_dataset_tamrl, batch_size=args.batch_size, shuffle=False, drop_last=False)

        forward_model, inverse_model = train_tamrl(forward_model, inverse_model, train_loader_tamrl, val_loader_tamrl, criterion, device, 
                                                   args.num_epochs, stride, optimizer, scheduler, patience)
        
        results = carbonbench.eval_tamrl_model(
            test_dataset_tamrl, test_cv, targets, forward_model, inverse_model, args.model, device, y_scaler,
            batch_size=args.batch_size
        )
    else:
        results = carbonbench.eval_nn_model(
            test_dataset, test_cv, targets, model, args.model, device, y_scaler,
            batch_size=args.batch_size
        )
    
    # Compute mean R2 across targets
    mean_r2 = np.mean([results[t]['R2'].mean() for t in targets])
    mean_rmse = np.mean([results[t]['RMSE'].mean() for t in targets])
    
    return mean_r2, mean_rmse, best_val_loss, epoch


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Split type: {args.split_type}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"{args.model}_{args.split_type}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    targets = ['GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50', 'NEE_VUT_USTAR50']
    y = carbonbench.load_targets(targets, qc=True)
    y_train, y_test = carbonbench.split_targets(
        y, task_type='zero-shot', split_type=args.split_type,
        verbose=True, plot=False
    )
    
    modis = carbonbench.load_modis()
    era = carbonbench.load_era('minimal')
    
    # Load CV split
    cv_split_path = f'./cv_5fold_{args.split_type}_split.json'
    with open(cv_split_path, 'r') as f:
        cv_split = json.load(f)
    
    # Get parameter grid
    param_grid = get_param_grid(args.model)
    print(f"\nParameter grid: {param_grid}")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Total runs: {len(all_combinations)} x 5 folds = {len(all_combinations) * 5}")
    
    # Grid search
    all_results = []
    
    for i, combo in enumerate(tqdm(all_combinations, desc="Grid Search")):
        hyperparams = dict(zip(param_names, combo))
        print(f"\n[{i+1}/{len(all_combinations)}] Testing: {hyperparams}")
        
        fold_r2_scores = []
        fold_rmse_scores = []
        fold_val_losses = []
        
        for fold in range(5):
            print(f"  Fold {fold}...")
            try:
                r2, rmse, val_loss, stopped_epoch = train_one_fold(
                    args, fold, hyperparams, y_train, modis, era, cv_split, device,
                    patience=args.patience
                )
                fold_r2_scores.append(r2)
                fold_rmse_scores.append(rmse)
                fold_val_losses.append(val_loss)
                print(f"(stopped at epoch {stopped_epoch}) R2={r2:.4f}, RMSE={rmse:.4f}")
            except Exception as e:
                print(f"Error: {e}")
                fold_r2_scores.append(np.nan)
                fold_rmse_scores.append(np.nan)
                fold_val_losses.append(np.nan)
        
        result = {
            **hyperparams,
            'cv_r2_mean': np.nanmean(fold_r2_scores),
            'cv_r2_std': np.nanstd(fold_r2_scores),
            'cv_rmse_mean': np.nanmean(fold_rmse_scores),
            'cv_rmse_std': np.nanstd(fold_rmse_scores),
            'cv_val_loss_mean': np.nanmean(fold_val_losses),
            'fold_r2_scores': fold_r2_scores,
        }
        all_results.append(result)
        
        print(f"  CV R2: {result['cv_r2_mean']:.4f} +/- {result['cv_r2_std']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('cv_r2_mean', ascending=False)
    results_path = os.path.join(output_path, 'gridsearch_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Find best parameters
    best_idx = results_df['cv_r2_mean'].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    
    best_params_clean = {k: v for k, v in best_params.items() 
                         if k in param_names}
    
    best_params_path = os.path.join(output_path, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_params': best_params_clean,
            'cv_r2_mean': float(best_params['cv_r2_mean']),
            'cv_r2_std': float(best_params['cv_r2_std']),
            'cv_rmse_mean': float(best_params['cv_rmse_mean']),
            'model': args.model,
            'split_type': args.split_type,
        }, f, indent=2)
    
    print("Grid Search Complete")
    print(f"Results saved to: {results_path}")
    print(f"Best params saved to: {best_params_path}")
    print(f"\nBest parameters:")
    for k, v in best_params_clean.items():
        print(f"  {k}: {v}")
    print(f"\nBest CV R2: {best_params['cv_r2_mean']:.4f} +/- {best_params['cv_r2_std']:.4f}")
    print(f"Best CV RMSE: {best_params['cv_rmse_mean']:.4f}")


if __name__ == '__main__':
    main()