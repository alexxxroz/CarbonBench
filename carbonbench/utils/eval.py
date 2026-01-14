import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Subset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from .processing import SlidingWindowDataset, SlidingWindowDatasetTAMRL

def mape(mean_flux: float, true: np.ndarray, pred: np.ndarray):
    # computes absolute error normalized by site mean flux
    mape = np.abs(pred - true) / (np.abs(mean_flux) + 1e-9)
    return np.mean(mape)

def eval_tree_model(X_test: pd.DataFrame, y_test: pd.DataFrame, targets: list, model: Any, y_scaler: StandardScaler, method: str='xgb'):
    # Evaluation of a tree-based model (XGBoost is default)
    res = {target: {'site': [], 'IGBP': [], 'Koppen': [], 'R2': [], 'RMSE': [], 'MAPE': []} for target in targets}
    for site in X_test.site.unique():
        X_site, y_site = X_test[X_test.site==site].drop('site', axis=1), y_test[X_test.site==site]
        if method=='xgb':
            X_site = xgb.DMatrix(X_site, enable_categorical=True)
        preds = y_scaler.inverse_transform(model.predict(X_site))
        y_site = y_scaler.inverse_transform(y_site)
        for idx, target in enumerate(targets):
            res[target]['site'].append(site)
            res[target]['IGBP'].append(X_test[X_test.site==site].IGBP.values[0])
            res[target]['Koppen'].append(X_test[X_test.site==site].Koppen.values[0])
            res[target]['R2'].append(np.clip(r2_score(y_site[:, idx], preds[:, idx]), 0, 1))
            res[target]['RMSE'].append(root_mean_squared_error(y_site[:, idx], preds[:, idx]))
            res[target]['MAPE'].append(mape(np.mean(y_site[:, idx]), y_site[:, idx], preds[:, idx]))
    print("\t\t\tR2\tRMSE\tMAPE")
    for target in targets:
        res[target] = pd.DataFrame(res[target])
        print(f"{target}:\t{round(np.mean(res[target]['R2']),2)}\t{round(np.mean(res[target]['RMSE']),2)}\t{round(np.mean(res[target]['MAPE']),2)}") 
    return res

def _eval_model_common(test_dataset, test, targets, predict_fn, y_scaler, batch_size=32):
    res = {target: {'site': [], 'IGBP': [], 'Koppen': [], 'R2': [], 'RMSE': [], 'MAPE': []} for target in targets}
    
    for site in test_dataset.get_sites():
        site_indices = test_dataset.get_site_indices(site)
        site_subset = Subset(test_dataset, site_indices)
        site_loader = DataLoader(site_subset, batch_size=batch_size, shuffle=False)

        preds, true = predict_fn(site_loader, site)
        if preds.ndim == 1:
            preds, true = preds.reshape(1, -1), true.reshape(1, -1)

        preds = y_scaler.inverse_transform(preds)#.reshape(-1, preds.shape[2]))
        y_site = y_scaler.inverse_transform(true)#.reshape(-1, true.shape[2]))
        
        site_meta = test[test.site == site].iloc[0]
        for idx, target in enumerate(targets):
            res[target]['site'].append(site)
            res[target]['IGBP'].append(site_meta.IGBP)
            res[target]['Koppen'].append(site_meta.Koppen)
            res[target]['R2'].append(np.clip(r2_score(y_site[:, idx], preds[:, idx]), 0, 1))
            res[target]['RMSE'].append(root_mean_squared_error(y_site[:, idx], preds[:, idx]))
            res[target]['MAPE'].append(mape(np.mean(y_site[:, idx]), y_site[:, idx], preds[:, idx]))
    
    print("\t\t\tR2\tRMSE\tMAPE")
    for target in targets:
        res[target] = pd.DataFrame(res[target])
        r = res[target]
        print(f"{target}:\t{r['R2'].mean():.2f}\t{r['RMSE'].mean():.2f}\t{r['MAPE'].mean():.2f}")
    return res


def eval_nn_model(test_dataset, test, targets, model, architecture, device, y_scaler, batch_size=32):
    def predict_fn(loader, site):
        return nn_predict(model, loader, architecture, device)
    return _eval_model_common(test_dataset, test, targets, predict_fn, y_scaler, batch_size)


def eval_tamrl_model(test_dataset, test, targets, forward_model, inverse_model, architecture, device, y_scaler, batch_size=32):
    def predict_fn(loader, site):
        historical = test_dataset.get_site_historical(site)
        return tamrl_predict(forward_model, inverse_model, loader, historical, architecture, device)
    return _eval_model_common(test_dataset, test, targets, predict_fn, y_scaler, batch_size)

def nn_predict(model: torch.nn.Module, test_loader: DataLoader, architecture: str, device: str):
    # Get prediction from a torch model using stride=1
    stride = 1
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for x, x_static, y, _, _, _ in test_loader:
            x, x_static, y = x.to(device), x_static.to(device), y.squeeze().to(device)
            if architecture in ['lstm', 'gru']:
                preds = model(x)
            else:
                preds = model(x, x_static)
            test_preds.append(preds[:,-stride,:].detach().cpu())
            
            y_cpu = y.detach().cpu()
            if y_cpu.dim() == 1:
                y_cpu = y_cpu.unsqueeze(0)   # restore batch dim
            test_true.append(y_cpu)
    test_preds = torch.cat(test_preds).squeeze()
    test_true = torch.cat(test_true).squeeze()
    return test_preds, test_true

def tamrl_predict(forward_model: torch.nn.Module, inverse_model: torch.nn.Module, test_loader: DataLoader, historical: tuple, architecture: str, device: str):
    # Get prediction from a torch model using stride=1
    stride = 1
    forward_model.eval()
    inverse_model.eval()
    test_preds = []
    test_true = []
    #site_hist_x, site_hist_cat = historical
    #site_hist_x, site_hist_cat = site_hist_x.to(device), site_hist_cat.to(device) 
    with torch.no_grad():
        for x, x_static, y, _, _, _, x_sup, x_static_sup in test_loader:
            x, x_static, y, x_sup, x_static_sup = x.to(device), x_static.to(device), y.squeeze().to(device), x_sup.to(device), x_static_sup.to(device)

            batch, window, _ = x.shape
            batch_dynamic_input = torch.cat((x, x_sup), dim=0)
            batch_static_input = torch.cat((x_static, x_static_sup), dim=0)

            batch_input = torch.cat((batch_dynamic_input, batch_static_input), dim=-1).to(device)
            latent_repr, _,_,_ = inverse_model(x=batch_input.float())

            batch_static_input = latent_repr[:x.shape[0]].unsqueeze(1).repeat(1, window, 1) # GET BATCH DATA FOR FORWARD MODEL
            pred = forward_model(x_dynamic=x.float().to(device), x_static=batch_static_input.float().to(device))
            if y.dim() == 1:
                y = y.unsqueeze(0)   # restore batch dim

            test_preds.append(pred[:,-stride,:].detach().cpu())
            test_true.append(y.detach().cpu())


    test_preds = torch.cat(test_preds).squeeze()
    test_true = torch.cat(test_true).squeeze()
    return test_preds, test_true

def plot_bars(results: dict, metrics: list, targets: list, save_path: str=""):
    # Produce barplot of average metrics across all test sites
    palette = [ "#543005", "#8c510a", "#bf812d", "#80cdc1", "#35978f", "#003c30"   ]
    models = list(results.keys())
    fig, axes = plt.subplots(len(metrics), len(targets), figsize=(16, 16), sharey='row')
    fig.suptitle(f"Average Site-level Results", fontsize=16)

    for i, metric in enumerate(metrics):
        for j, target in enumerate(targets):
            ax = axes[i, j] if len(metrics) > 1 else axes[j]

            plot_data = []
            for model in models:
                df = results[model][target]

                agg = pd.DataFrame({
                    'model': [model],
                    'mean': [df[metric].mean()],
                    'std': [df[metric].std()]
                })
                plot_data.append(agg)

            plot_df = pd.concat(plot_data)

            x_pos = np.arange(len(plot_df))
            ax.bar(x_pos, plot_df['mean'], color=palette[:len(plot_df)], 
                   yerr=plot_df['std'], capsize=5, error_kw={'elinewidth': 0.5, 'ecolor': 'black'})
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_df['model'], rotation=0)
            
            if metric in ['RMSE', 'MAPE']: # find best model
                best_idx = plot_df.groupby('model')['mean'].mean().idxmin()
            else:
                best_idx = plot_df.groupby('model')['mean'].mean().idxmax()

            # highlight best model
            for idx, (bar, mdl) in enumerate(zip(ax.patches, plot_df['model'].values)):
                if mdl == best_idx:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)

            ax.set_title(target if i == 0 else "")
            if j == 0:
                ax.set_ylabel(metric)
            else:
                ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    if len(save_path) > 0:
        plt.savefig(os.join(save_path, "barplot.png"))
    plt.show()

def plot_heatmap(results: dict, metrics: list, targets: list, classification: str='IGBP', save_path: str=""):
    # Creates a figure with metrics averaged by the selected classification groups
    koppen_map = {
        'A': 'Tropical',
        'B': 'Arid',
        'C': 'Temperate',
        'D': 'Continental',
        'E': 'Polar'
    }
    models = list(results.keys())
    fig, axes = plt.subplots(len(metrics), len(targets), figsize=(16, 16))
    fig.suptitle(f"{classification} Classification", fontsize=16)
    
    for i, metric in enumerate(metrics):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            
            heatmap_data = []
            for model in models:
                df = results[model][target]
                grouped = df.groupby(classification)[metric].mean()
                grouped.name = model
                heatmap_data.append(grouped)
            
            pivot_df = pd.concat(heatmap_data, axis=1)
            
            if classification == 'Koppen':
                pivot_df.index = pivot_df.index.map(lambda x: koppen_map.get(x[0], x))
            
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='BrBG', 
                        ax=ax)
            ax.set_title(target if i == 0 else "")
            ax.set_ylabel(metric if j == 0 else "")
            ax.set_xlabel(classification if i == len(metrics)-1 else "")
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    if len(save_path) > 0:
        plt.savefig(os.join(save_path, f"{classification}_heatmap.png"))
    plt.show()
    

