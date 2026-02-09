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

from utils import train_tamrl


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='CarbonBench Ensemble Model Training')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'ctlstm', 'gru', 'ctgru', 'transformer', 'patch_transformer', 'tam-rl', 'mlp'],
                        help='Model architecture')
    parser.add_argument('--split_type', type=str, required=True,
                        choices=['IGBP', 'Koppen'],
                        help='Train-test split stratification type')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file with best hyperparameters')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=27,
                        help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
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
        'tam-rl': carbonbench.lstm,
        'mlp': carbonbench.mlp,
    }
    return model_map[model_name](**kwargs)


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    config = load_config(args.config)
    best_params = config['best_params']
    
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Split type: {args.split_type}")
    print(f"Config file: {args.config}")
    print(f"Best params from grid search: {best_params}")
    
    output_subdir = f"{args.model}_seed{args.seed}_{args.split_type}"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    print(f"Output: {output_path}")
    
    print("\nLoading data...")
    targets = ['GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50', 'NEE_VUT_USTAR50']
    include_qc = True
    test_QC_threshold = 1
    
    y = carbonbench.load_targets(targets, include_qc)
    y_train, y_test = carbonbench.split_targets(
        y, task_type='zero-shot', split_type=args.split_type, 
        verbose=True, plot=False
    )
    
    modis = carbonbench.load_modis()
    era = carbonbench.load_era('minimal')
    
    train, val, _, x_scaler, y_scaler = carbonbench.join_features(
        y_train, y_test, modis, era, val_ratio=0.2, scale=True
    )
    
    print("\nCreating data loaders...")
    batch_size = args.batch_size
    window_size = 30
    stride = 15 if args.model != 'mlp' else 1
    num_workers = 4
    
    train_hist = carbonbench.historical_cache(train, era, modis, x_scaler, window_size)
    train_dataset = carbonbench.SlidingWindowDataset(
        train_hist, targets, include_qc,
        window_size=window_size, stride=stride,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=False
    )
    
    val_hist = carbonbench.historical_cache(val, era, modis, x_scaler, window_size)
    val_dataset = carbonbench.SlidingWindowDataset(
        val_hist, targets, include_qc,
        window_size=window_size, stride=stride,
        encoders=train_dataset.encoders,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    x_sample, x_static_sample, _, _, _, _ = next(iter(train_loader))
    input_dynamic_channels = x_sample.shape[2]
    input_static_channels = x_static_sample.shape[2]
    output_channels = len(targets)
    
    print(f"Dynamic channels: {input_dynamic_channels}, Static channels: {input_static_channels}")
    
    print("\nInitializing model...")
    
    IGBP = train['IGBP'].values
    IGBP_weights = compute_class_weight(class_weight="balanced", classes=np.unique(IGBP), y=IGBP)
    IGBP_weights = {str(k): float(IGBP_weights[i]) for i, k in enumerate(np.unique(IGBP))}
    
    Koppen = train['Koppen'].values
    Koppen_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Koppen), y=Koppen)
    Koppen_weights = {str(k): float(Koppen_weights[i]) for i, k in enumerate(np.unique(Koppen))}
    
    if args.model == 'mlp':
        model_kwargs = {
            'input_dim': input_dynamic_channels + input_static_channels,
            'hidden_dim': best_params['hidden_dim'],
            'output_dim': output_channels,
            'dropout': best_params['dropout'],
        }
    else:
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

        if args.model == 'tam-rl':
            model_kwargs['latent_dim'] = best_params['latent_dim']

    model = get_model(args.model, **model_kwargs).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    criterion = carbonbench.CustomLoss(IGBP_weights, Koppen_weights, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"\nTraining for {args.num_epochs} epochs with learning rate {best_params['lr']}...")
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Training"):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for x, x_static, y_batch, qc, igbp_w, koppen_w in train_loader:
            x = x.to(device)
            x_static = x_static.to(device)
            y_batch = y_batch.to(device)
            qc = qc.to(device)
            igbp_w = igbp_w.to(device)
            koppen_w = koppen_w.to(device)
            
            optimizer.zero_grad()

            if args.model == 'mlp':
                x_cat = torch.cat([x[:, -1, :], x_static[:, -1, :]], dim=-1)
                pred = model(x_cat).unsqueeze(1)
                loss = criterion(pred, y_batch[:, -1:, :3], qc, igbp_w, koppen_w)
            elif args.model in ['lstm', 'gru', 'tam-rl']:
                pred = model(x)
                loss = criterion(pred[:, -stride:, :], y_batch[:, -stride:, :3], qc, igbp_w, koppen_w)
            else:
                pred = model(x, x_static)
                loss = criterion(pred[:, -stride:, :], y_batch[:, -stride:, :3], qc, igbp_w, koppen_w)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        train_loss = epoch_loss / num_batches
        train_losses.append(train_loss)
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for x, x_static, y_batch, _, _, _ in val_loader:
                    x = x.to(device)
                    x_static = x_static.to(device)
                    y_batch = y_batch.to(device)
                    
                    if args.model == 'mlp':
                        x_cat = torch.cat([x[:, -1, :], x_static[:, -1, :]], dim=-1)
                        pred = model(x_cat).unsqueeze(1)
                    elif args.model in ['lstm', 'gru', 'tam-rl']:
                        pred = model(x)
                    else:
                        pred = model(x, x_static)

                    val_preds.append(pred.cpu())
                    val_true.append(y_batch.cpu())

            val_preds = torch.cat(val_preds)
            val_true = torch.cat(val_true)
            if args.model == 'mlp':
                val_loss = criterion(
                    val_preds.to(device),
                    val_true[:, -1:, :3].to(device)
                ).item()
            else:
                val_loss = criterion(
                    val_preds[:, -stride:, :].to(device),
                    val_true[:, -stride:, :3].to(device)
                ).item()
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nTraining complete, best val loss: {best_val_loss:.4f}")

    if args.model == 'tam-rl':
        inverse_model = carbonbench.ae_tamrl(
            input_channels=model_kwargs['input_dynamic_channels'] + model_kwargs['input_static_channels'],
            code_dim=model_kwargs['latent_dim'],
            hidden_dim=model_kwargs['latent_dim'],
            output_channels=model_kwargs['latent_dim']
        ).to(device)
        forward_model = carbonbench.tamlstm(
            model_kwargs['input_dynamic_channels'],
            model_kwargs['latent_dim'],
            model_kwargs['hidden_dim'],
            model_kwargs['output_channels'],
            model_kwargs['dropout'],
            model_kwargs['layers']
        ).to(device)

        encoder_weights = {k.replace('encoder.', ''): v for k, v in best_model_state.items() if k.startswith('encoder.')}
        forward_model.encoder.load_state_dict(encoder_weights)

        criterion_tamrl = carbonbench.CustomLoss(IGBP_weights, Koppen_weights)
        optimizer_tamrl = optim.Adam(list(inverse_model.parameters()) + list(forward_model.parameters()), lr=1e-3)
        scheduler_tamrl = StepLR(optimizer_tamrl, step_size=10, gamma=0.5)

        train_dataset_tamrl = carbonbench.SlidingWindowDatasetTAMRL(
            train_hist, targets, include_qc,
            window_size=window_size, stride=stride,
            cat_features=['IGBP', 'Koppen', 'Koppen_short']
        )
        train_loader_tamrl = DataLoader(train_dataset_tamrl, batch_size=batch_size, shuffle=True)

        val_dataset_tamrl = carbonbench.SlidingWindowDatasetTAMRL(
            val_hist, targets, include_qc,
            window_size=window_size, stride=stride,
            encoders=train_dataset.encoders,
            cat_features=['IGBP', 'Koppen', 'Koppen_short']
        )
        val_loader_tamrl = DataLoader(val_dataset_tamrl, batch_size=batch_size, shuffle=True)

        print("\nTraining TAM-RL phase...")
        forward_model, inverse_model = train_tamrl(
            forward_model, inverse_model, train_loader_tamrl, val_loader_tamrl,
            criterion_tamrl, device, args.num_epochs, stride, optimizer_tamrl, scheduler_tamrl
        )

        checkpoint_path = os.path.join(output_path, 'forward_model.pt')
        torch.save(forward_model.state_dict(), checkpoint_path)
        print(f"Forward model saved to: {checkpoint_path}")

        checkpoint_path = os.path.join(output_path, 'inverse_model.pt')
        torch.save(inverse_model.state_dict(), checkpoint_path)
        print(f"Inverse model saved to: {checkpoint_path}")
    else:
        checkpoint_path = os.path.join(output_path, 'model.pt')
        torch.save(best_model_state, checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")
    
    print("\nDone")


if __name__ == '__main__':
    main()