# # Train one model with a given seed; optionally run 5-fold CV with a fold id.

import os
import sys
import json
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


def parse_args():
    parser = argparse.ArgumentParser(description='CarbonBench Single Model Training')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'ctlstm', 'gru', 'ctgru', 'transformer', 'patch_transformer'],
                        help='Model architecture')
    parser.add_argument('--split_type', type=str, required=True,
                        choices=['IGBP', 'Koppen'],
                        help='Train-test split stratification type')
    parser.add_argument('--fold', type=int, default=None,
                        choices=[0, 1, 2, 3, 4],
                        help='CV fold number (0-4). If not specified, runs normal train/test mode.')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=27,
                        help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-4 for transformer/patch_transformer, 1e-3 for others)')
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
    
    set_seed(args.seed)
    
    if args.lr is None:
        if args.model in ['transformer', 'patch_transformer']:
            args.lr = 1e-4
        else:
            args.lr = 1e-3
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # determine mode
    cv_mode = args.fold is not None
    mode_str = f"CV fold {args.fold}" if cv_mode else "Normal train/test"
    
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Split type: {args.split_type}")
    print(f"Learning rate: {args.lr}")
    print(f"Mode: {mode_str}")
    
    if cv_mode:
        output_subdir = f"{args.model}_seed{args.seed}_{args.split_type}_fold{args.fold}"
    else:
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
    
    # Different data preparation for CV mode vs normal mode
    if cv_mode:
        # Split training set by sites according to fold
        cv_split_path = f'./cv_5fold_{args.split_type}_split.json'
        
        if not os.path.exists(cv_split_path):
            raise FileNotFoundError(
                f"CV split file not found: {cv_split_path}\n"
                f"Please run the CV split generation script first."
            )
        
        with open(cv_split_path, 'r') as f:
            cv_split = json.load(f)
        
        test_sites = cv_split['folds'][str(args.fold)]
        print(f"\nCV Mode: Fold {args.fold}")
        
        # Split y_train into CV train and CV val by sites
        y_train_cv = y_train[~y_train.site.isin(test_sites)]
        y_test_cv = y_train[y_train.site.isin(test_sites)]
        
        # join_features will return (train_split, val_internal, cv_val_as_test, ...)
        # I merge train_split + val_internal to get full CV training data
        train_cv, val_cv, test_cv, x_scaler, y_scaler = carbonbench.join_features(
            y_train_cv, y_test_cv, modis, era, val_ratio=0.2, scale=True
        )
        
        # Combine train_temp and val_internal as actual CV training set
        # TODO: WHY?
#         train = pd.concat([train_cv, val_cv]).reset_index(drop=True)
#         test = test_cv  # This is our CV validation set (the held-out fold sites)
#         test = None   # No test set in CV mode
        
        print(f"\nCV Data sizes:")
        print(f"  Train samples: {len(train_cv)}")
        print(f"  Val samples: {len(val_cv)}")
        print(f"  Test samples: {len(test_cv)}")
        
    else:
        train_cv, val_cv, test_cv, x_scaler, y_scaler = carbonbench.join_features(
            y_train, y_test, modis, era, val_ratio=0.2, scale=True
        )
        print(f"\nData sizes:")
        print(f"  Train samples: {len(train_cv)}")
        print(f"  Val samples: {len(val_cv)}")
        print(f"  Test samples: {len(test_cv)}")
    
    print("\nCreating data loaders...")
    batch_size = args.batch_size
    window_size = 30
    stride = 15
    num_workers = 4
    
    # Train loader
    train_hist = carbonbench.historical_cache(train_cv, era, modis, x_scaler, window_size)
    train_dataset = carbonbench.SlidingWindowDataset(
        train_hist, targets, include_qc,
        window_size=window_size, stride=stride,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=False # TODO: Why was it True?
    )
    
    # Val loader 
    val_hist = carbonbench.historical_cache(val_cv, era, modis, x_scaler, window_size)
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
    
    # Evaluation dataset (stride=1)
    if cv_mode:
        # CV mode: create evaluation dataset from val data with stride=1
        # TODO: why do you pass the same dataset to the eval dataset?
        test_hist = carbonbench.historical_cache(test_cv, era, modis, x_scaler, window_size)
        test_dataset = carbonbench.SlidingWindowDataset(
            test_hist, targets, include_qc,
            window_size=window_size, stride=1, QC_threshold=test_QC_threshold,
            encoders=train_dataset.encoders,
            cat_features=['IGBP', 'Koppen', 'Koppen_short']
        )
    else:
        # normal mode: create test dataset with stride=1
        test_hist = carbonbench.historical_cache(test_cv, era, modis, x_scaler, window_size)
        test_dataset = carbonbench.SlidingWindowDataset(
            test_hist, targets, include_qc,
            window_size=window_size, stride=1, QC_threshold=test_QC_threshold,
            encoders=train_dataset.encoders,
            cat_features=['IGBP', 'Koppen', 'Koppen_short']
        )
    
    # dimensions
    x_sample, x_static_sample, _, _, _, _ = next(iter(train_loader))
    input_dynamic_channels = x_sample.shape[2]
    input_static_channels = x_static_sample.shape[2]
    output_channels = len(targets)
    
    print(f"Dynamic channels: {input_dynamic_channels}, Static channels: {input_static_channels}")
    

    print("\nInitializing model...")
    
    IGBP = train_cv['IGBP'].values
    IGBP_weights = compute_class_weight(class_weight="balanced", classes=np.unique(IGBP), y=IGBP)
    IGBP_weights = {str(k): float(IGBP_weights[i]) for i, k in enumerate(np.unique(IGBP))}
    
    Koppen = train_cv['Koppen'].values
    Koppen_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Koppen), y=Koppen)
    Koppen_weights = {str(k): float(Koppen_weights[i]) for i, k in enumerate(np.unique(Koppen))}
    
    # Model hyperparameters
    hidden_dim = 128
    dropout = 0.2
    layers = 2
    nhead = 4
    tf_layers = 3
    patch_len = 10
    patch_stride = 5
    
    # Build model arguments based on architecture
    model_kwargs = {
        'input_dynamic_channels': input_dynamic_channels,
        'hidden_dim': hidden_dim,
        'output_channels': output_channels,
        'dropout': dropout,
    }
    
    if args.model in ['lstm', 'gru']:
        model_kwargs['layers'] = layers
    else:
        model_kwargs['input_static_channels'] = input_static_channels
    
    if 'transformer' in args.model:
        model_kwargs['nhead'] = nhead
        model_kwargs['num_layers'] = tf_layers
        model_kwargs['seq_len'] = window_size
        if 'patch' in args.model:
            model_kwargs['pred_len'] = stride
            model_kwargs['patch_len'] = patch_len
            model_kwargs['stride'] = patch_stride
    else:
        model_kwargs['layers'] = layers
    
    model = get_model(args.model, **model_kwargs).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = carbonbench.CustomLoss(IGBP_weights, Koppen_weights, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"\nTraining for {args.num_epochs} epochs...")
    
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
            
            if args.model in ['lstm', 'gru']:
                pred = model(x)
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
        
        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for x, x_static, y_batch, _, _, _ in val_loader:
                    x = x.to(device)
                    x_static = x_static.to(device)
                    y_batch = y_batch.to(device)
                    
                    if args.model in ['lstm', 'gru']:
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
            val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nTraining complete, best val loss: {best_val_loss:.4f}")
    
    # Save model checkpoint
    checkpoint_path = os.path.join(output_path, 'model.pt')
    torch.save(best_model_state, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")
    
    # evaluations
    if cv_mode:
        print(f"\nEvaluating on CV validation set (fold {args.fold})...")
    else:
        print("\nEvaluating on test set...")
    
    results = carbonbench.eval_nn_model(
        test_dataset, test_cv, targets, model, args.model, device, y_scaler,
        batch_size=batch_size, window_size=window_size
    )
    
    # Save results
    results_data = []
    for target in targets:
        df = results[target]
        result_row = {
            'model': args.model,
            'seed': args.seed,
            'split_type': args.split_type,
            'target': target,
            'R2_mean': df['R2'].mean(),
            'R2_std': df['R2'].std(),
            'RMSE_mean': df['RMSE'].mean(),
            'RMSE_std': df['RMSE'].std(),
            'MAPE_mean': df['MAPE'].mean(),
            'MAPE_std': df['MAPE'].std(),
        }
        if cv_mode:
            result_row['fold'] = args.fold
        results_data.append(result_row)
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(output_path, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    # Save per-site predictions
    for target in targets:
        site_results_path = os.path.join(output_path, f'site_results_{target}.csv')
        results[target].to_csv(site_results_path, index=False)
    
    # summary
    print(f"\n{'='*60}")
    print(f"Summary: {args.model} | seed={args.seed} | split={args.split_type} | lr={args.lr}", end="")
    if cv_mode:
        print(f" | fold={args.fold}")
    else:
        print()
    print(f"{'='*60}")
    
    for target in targets:
        df = results[target]
        print(f"{target}: R2={df['R2'].mean():.3f}, RMSE={df['RMSE'].mean():.3f}, MAPE={df['MAPE'].mean():.3f}")
    
    print("\nDone")


if __name__ == '__main__':
    main()

