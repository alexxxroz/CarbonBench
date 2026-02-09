# Evaluate tam-rl ensemble models by averaging predictions across 10 seeds

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
    parser = argparse.ArgumentParser(description='Evaluate tam-rl ensemble models on test set')
    parser.add_argument('--split_type', type=str, required=True,
                        choices=['IGBP', 'Koppen'],
                        help='Split type')
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


def mape(mean_flux, true, pred):
    mape_val = np.abs(pred - true) / (np.abs(mean_flux) + 1e-9)
    return np.mean(mape_val)


def get_predictions_tamrl(forward_model, inverse_model, loader, device):
    """Get predictions from a single tam-rl model pair for all samples in loader."""
    forward_model.eval()
    inverse_model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, x_static, y, _, _, _, x_sup, x_static_sup in loader:
            x = x.to(device)
            x_static = x_static.to(device)
            x_sup = x_sup.to(device)
            x_static_sup = x_static_sup.to(device)

            batch, window, _ = x.shape
            batch_dynamic_input = torch.cat((x, x_sup), dim=0)
            batch_static_input = torch.cat((x_static, x_static_sup), dim=0)

            batch_input = torch.cat((batch_dynamic_input, batch_static_input), dim=-1).to(device)
            latent_repr, _, _, _ = inverse_model(x=batch_input.float())

            batch_static_input = latent_repr[:x.shape[0]].unsqueeze(1).repeat(1, window, 1)
            pred = forward_model(x_dynamic=x.float(), x_static=batch_static_input.float())

            all_preds.append(pred[:, -1, :].cpu().numpy())

            y_cpu = y[:, -1, :].cpu().numpy()
            all_true.append(y_cpu)

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    return all_preds, all_true


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: tam-rl, Split: {args.split_type}")

    config = load_config(args.config)
    best_params = config['best_params']
    print(f"Best params: {best_params}")

    seeds = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

    available_seeds = []
    for seed in seeds:
        model_subdir = f"tam-rl_seed{seed}_{args.split_type}"
        forward_path = os.path.join(args.model_dir, model_subdir, 'forward_model.pt')
        inverse_path = os.path.join(args.model_dir, model_subdir, 'inverse_model.pt')
        if os.path.exists(forward_path) and os.path.exists(inverse_path):
            available_seeds.append(seed)
        else:
            print(f"Warning: Model not found for seed {seed}")

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
    y_train, y_test = carbonbench.split_targets(
        y, task_type='zero-shot', split_type=args.split_type,
        verbose=False, plot=False
    )

    modis = carbonbench.load_modis()
    era = carbonbench.load_era('minimal')

    train_cv, val_cv, test_cv, x_scaler, y_scaler = carbonbench.join_features(
        y_train, y_test, modis, era, val_ratio=0.2, scale=True
    )

    train_hist = carbonbench.historical_cache(train_cv, era, modis, x_scaler, window_size)
    train_dataset = carbonbench.SlidingWindowDataset(
        train_hist, targets, include_qc,
        window_size=window_size, stride=stride,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )

    test_hist = carbonbench.historical_cache(test_cv, era, modis, x_scaler, window_size)
    test_dataset = carbonbench.SlidingWindowDatasetTAMRL(
        test_hist, targets, include_qc,
        window_size=window_size, stride=1, QC_threshold=test_QC_threshold,
        encoders=train_dataset.encoders,
        cat_features=['IGBP', 'Koppen', 'Koppen_short']
    )

    # Get dimensions from train loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    x_sample, x_static_sample, _, _, _, _ = next(iter(train_loader))
    input_dynamic_channels = x_sample.shape[2]
    input_static_channels = x_static_sample.shape[2]
    output_channels = len(targets)

    print(f"Dynamic channels: {input_dynamic_channels}, Static channels: {input_static_channels}")

    latent_dim = best_params['latent_dim']

    print(f"\nEvaluating ensemble on test set...")

    results = {target: {'site': [], 'IGBP': [], 'Koppen': [],
                        'R2': [], 'RMSE': [], 'MAPE': [],
                        'pred_std': []} for target in targets}

    for site in test_dataset.get_sites():
        site_indices = test_dataset.get_site_indices(site)
        site_subset = Subset(test_dataset, site_indices)
        site_loader = DataLoader(site_subset, batch_size=args.batch_size, shuffle=False)

        all_seed_preds = []
        true_values = None

        for seed in available_seeds:
            model_subdir = f"tam-rl_seed{seed}_{args.split_type}"
            forward_path = os.path.join(args.model_dir, model_subdir, 'forward_model.pt')
            inverse_path = os.path.join(args.model_dir, model_subdir, 'inverse_model.pt')

            forward_model = carbonbench.tamlstm(
                input_dynamic_channels,
                latent_dim,
                best_params['hidden_dim'],
                output_channels,
                best_params['dropout'],
                best_params['layers']
            ).to(device)

            inverse_model = carbonbench.ae_tamrl(
                input_channels=input_dynamic_channels + input_static_channels,
                code_dim=latent_dim,
                hidden_dim=latent_dim,
                output_channels=latent_dim
            ).to(device)

            forward_model.load_state_dict(torch.load(forward_path, map_location=device))
            inverse_model.load_state_dict(torch.load(inverse_path, map_location=device))

            preds, true = get_predictions_tamrl(forward_model, inverse_model, site_loader, device)
            all_seed_preds.append(preds)

            if true_values is None:
                true_values = true

        all_seed_preds = np.stack(all_seed_preds, axis=0)
        ensemble_preds = np.mean(all_seed_preds, axis=0)
        pred_std = np.std(all_seed_preds, axis=0)

        ensemble_preds = y_scaler.inverse_transform(ensemble_preds)
        true_values = y_scaler.inverse_transform(true_values)
        pred_std_orig = pred_std * y_scaler.scale_

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
        df['model'] = 'tam-rl'
        df['split_type'] = args.split_type
        df['n_seeds'] = len(available_seeds)
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)

    output_file = os.path.join(args.output_dir, f"tam-rl_{args.split_type}_ensemble.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()