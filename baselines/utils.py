import numpy as np

import torch
from tqdm import tqdm

def train_tamrl(forward_model, inverse_model, train_loader_tamrl, val_loader_tamrl, criterion, device, num_epoch, stride, optimizer, scheduler, patience=10):
    '''Training TAM-RL'''
    best = np.inf
    no_improve_count = 0
    for epoch in tqdm(range(num_epoch)):
        inverse_model.train()
        forward_model.train()
        for x, x_static, y, qc, igbp_w, koppen_w, x_sup, x_static_sup in train_loader_tamrl:
            x, x_static, y, qc, igbp_w, koppen_w, x_sup, x_static_sup = x.to(device), x_static.to(device), y.to(device), qc.to(device),\
                                                                        igbp_w.to(device), koppen_w.to(device), x_sup.to(device), x_static_sup.to(device)
            optimizer.zero_grad()

            batch, window, _ = x.shape
            batch_dynamic_input = torch.cat((x, x_sup), dim=0)
            batch_static_input = torch.cat((x_static, x_static_sup), dim=0)

            batch_input = torch.cat((batch_dynamic_input, batch_static_input), dim=-1).to(device)
            latent_repr, _,_,_ = inverse_model(x=batch_input.float())

            batch_static_input = latent_repr[:x.shape[0]].unsqueeze(1).repeat(1, window, 1) # GET BATCH DATA FOR FORWARD MODEL
            pred = forward_model(x_dynamic=x.float().to(device), x_static=batch_static_input.float().to(device))

            if criterion.__class__.__name__=='CustomLoss':
                error = criterion(pred[:,-stride:, :], y[:,-stride:,:3], qc, igbp_w, koppen_w)
            else:
                error = criterion(pred[:,-stride:, :], y[:,-stride:,:3])

            error.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            inverse_model.eval()
            forward_model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for x, x_static, y, _, _, _, x_sup, x_static_sup in val_loader_tamrl:
                    x, x_static, y, x_sup, x_static_sup = x.to(device), x_static.to(device), y.squeeze().to(device), x_sup.to(device), x_static_sup.to(device)

                    batch, window, _ = x.shape
                    batch_dynamic_input = torch.cat((x, x_sup), dim=0)
                    batch_static_input = torch.cat((x_static, x_static_sup), dim=0)

                    batch_input = torch.cat((batch_dynamic_input, batch_static_input), dim=-1).to(device)
                    latent_repr, _,_,_ = inverse_model(x=batch_input.float())

                    batch_static_input = latent_repr[:x.shape[0]].unsqueeze(1).repeat(1, window, 1) # GET BATCH DATA FOR FORWARD MODEL
                    pred = forward_model(x_dynamic=x.float().to(device), x_static=batch_static_input.float().to(device))

                    val_preds.append(pred.detach().cpu())
                    val_true.append(y.detach().cpu())
            val_preds = torch.cat(val_preds).squeeze()
            val_true = torch.cat(val_true).squeeze()

            #val_loss = criterion(val_preds.to(device)[:,-1:,:], val_true[:, :3].to(device))
            val_loss = criterion(val_preds.to(device)[:,-stride:,:], val_true[:,-stride:, :3].to(device))

            best_old = best
            best = min(val_loss, best)
            if best < best_old:
                best_model_fw = forward_model.state_dict()
                best_model_inv = inverse_model.state_dict()
                no_improve_count = 0  
            else:
                no_improve_count += 5 # since validated every 5 epochs
            
            if no_improve_count >= patience:
                break
    forward_model.load_state_dict(best_model_fw)
    inverse_model.load_state_dict(best_model_inv)
    return forward_model, inverse_model