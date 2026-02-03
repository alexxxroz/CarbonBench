'''Fine-tune the existing zero-shot models to make it few-shot'''

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

import carbonbench


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='CarbonBench Ensemble Model Training')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'ctlstm', 'gru', 'ctgru', 'transformer', 'patch_transformer'],
                        help='Model architecture')
    parser.add_argument('--split_type', type=str, required=True,
                        choices=['IGBP', 'Koppen'],
                        help='Train-test split stratification type')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file with best hyperparameters')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for results')
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
    }
    return model_map[model_name](**kwargs)


def main():
    args = parse_args()
    
    set_seed(27)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    config = load_config(args.config)
    best_params = config['best_params']
    
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Split type: {args.split_type}")
    print(f"Config file: {args.config}")
    
    print("\nLoading data...")
    targets = ['GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50', 'NEE_VUT_USTAR50']
    include_qc = True
    test_QC_threshold = 1
    
    y = carbonbench.load_targets(targets, include_qc)
    y_train, y_test, y_finetune = carbonbench.split_targets(
        y, task_type='few-shot', split_type=args.split_type, 
        verbose=False, plot=False
    )
    
    modis = carbonbench.load_modis()
    era = carbonbench.load_era('minimal')
    
    train, val, finetune, test, x_scaler, y_scaler = carbonbench.join_features_finetune(y_train, y_finetune, y_test, modis, era, scale=True)

    batch_size = args.batch_size
    window_size = 30
    stride = 15
    num_workers = 4
    
    train_hist = carbonbench.historical_cache(train, era, modis, x_scaler, window_size)
    train_dataset = carbonbench.SlidingWindowDataset(train_hist, targets, include_qc, window_size=window_size, stride=stride, cat_features=['IGBP', 'Koppen', 'Koppen_short'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_hist = carbonbench.historical_cache(val, era, modis, x_scaler, window_size)
    val_dataset = carbonbench.SlidingWindowDataset(val_hist, targets, include_qc, window_size=window_size, stride=stride, encoders=train_dataset.encoders,
                                                   cat_features=['IGBP', 'Koppen', 'Koppen_short'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    finetune_hist = carbonbench.historical_cache(finetune, era, modis, x_scaler, window_size)
    finetune_dataset = carbonbench.SlidingWindowDataset(finetune_hist, targets, include_qc, window_size=window_size, stride=1, encoders=train_dataset.encoders,
                                                        cat_features=['IGBP', 'Koppen', 'Koppen_short'])

    test_hist = carbonbench.historical_cache(test, era, modis, x_scaler, window_size)
    test_dataset = carbonbench.SlidingWindowDataset(test_hist, targets, include_qc, window_size=window_size, QC_threshold=test_QC_threshold, stride=1,
                                                    cat_features=['IGBP', 'Koppen', 'Koppen_short'],
                                        encoders=train_dataset.encoders)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    x_sample, x_static_sample, _, _, _, _ = next(iter(train_loader))
    input_dynamic_channels = x_sample.shape[2]
    input_static_channels = x_static_sample.shape[2]
    output_channels = len(targets)
    
    IGBP = train['IGBP'].values
    IGBP_weights = compute_class_weight(class_weight="balanced", classes=np.unique(IGBP), y=IGBP)
    IGBP_weights = {str(k): float(IGBP_weights[i]) for i, k in enumerate(np.unique(IGBP))}
    
    Koppen = train['Koppen'].values
    Koppen_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Koppen), y=Koppen)
    Koppen_weights = {str(k): float(Koppen_weights[i]) for i, k in enumerate(np.unique(Koppen))}
    
    '''Loading the models'''
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
            model_kwargs['patch_len'] = 4
            model_kwargs['stride'] = 4
    else:
        model_kwargs['layers'] = best_params['layers']
    
    
    seeds = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
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
    
    models = []
    for seed in available_seeds:
        model_subdir = f"{args.model}_seed{seed}_{args.split_type}"
        model_path = os.path.join(args.model_dir, model_subdir, 'model.pt')

        model = get_model(args.model, **model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        models.append(model)
        
    criterion = carbonbench.CustomLoss(IGBP_weights, Koppen_weights, device=device)
    
    
    stride_fine_tune = 1
    
    summary = {idx: [] for idx in seeds}
    for site in finetune_dataset.get_sites():
        site_indices = finetune_dataset.get_site_indices(site)
        site_subset = Subset(finetune_dataset, site_indices)
        site_loader = DataLoader(site_subset, batch_size=10_000, shuffle=False)
        
        for idx, model in enumerate(models):
            model.load_state_dict(torch.load(f'ctlstm.pth', map_location=device))
            optimizer = optim.Adam(model.parameters(), lr=5e-6) 
            res = {target: {'site': [], 'IGBP': [], 'Koppen': [], 'R2': [], 'RMSE': [], 'MAPE': []} for target in targets}
            '''Fine-tune on a site'''
            for epoch in range(15):
                model.train()
                for x, x_static, y, qc, igbp_w, koppen_w in site_loader:
                    x, x_static, y, qc, igbp_w, koppen_w = x.to(device), x_static.to(device), y.to(device), qc.to(device), igbp_w.to(device), koppen_w.to(device)
                    optimizer.zero_grad()
                    if architecture in ['lstm', 'gru']:
                        pred = model(x)
                    else:
                        pred = model(x, x_static)

                    if criterion.__class__.__name__=='CustomLoss':
                        error = criterion(pred[:,-stride_fine_tune:, :], y[:,-stride_fine_tune:,:3], qc, igbp_w, koppen_w)
                    else:
                        error = criterion(pred[:,-stride_fine_tune:, :], y[:,-stride_fine_tune:,:3])

                    error.backward()
                    optimizer.step()

            '''Test the site'''
            site_indices = test_dataset.get_site_indices(site)
            site_subset = Subset(test_dataset, site_indices)
            site_loader = DataLoader(site_subset, batch_size=batch_size, shuffle=False)
            if len(site_loader)>0:
                preds, true = carbonbench.nn_predict(model, site_loader, architecture, device)
                if preds.ndim == 1:
                    preds, true = preds.reshape(1, -1), true.reshape(1, -1)

                preds = y_scaler.inverse_transform(preds)
                y_site = y_scaler.inverse_transform(true)

                res = carbonbench.append_results(res, test, site, y_site, preds, targets)
                summary[seeds[idx]].append(res)
        print(summary[seeds[idx]])
        print("\nDone")


if __name__ == '__main__':
    main()