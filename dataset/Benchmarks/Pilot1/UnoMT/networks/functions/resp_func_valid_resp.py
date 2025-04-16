def valid_resp(device: torch.device, resp_net: nn.Module, data_loaders:
    torch.utils.data.DataLoader):
    resp_net.eval()
    mse_list = []
    mae_list = []
    r2_list = []
    print('\tDrug Response Regression:')
    with torch.no_grad():
        for val_loader in data_loaders:
            mse, mae = 0.0, 0.0
            growth_array, pred_array = np.array([]), np.array([])
            for rnaseq, drug_feature, conc, grth in val_loader:
                rnaseq, drug_feature, conc, grth = rnaseq.to(device
                    ), drug_feature.to(device), conc.to(device), grth.to(device
                    )
                pred_growth = resp_net(rnaseq, drug_feature, conc)
                num_samples = conc.shape[0]
                mse += F.mse_loss(pred_growth, grth).item() * num_samples
                mae += F.l1_loss(pred_growth, grth).item() * num_samples
                growth_array = np.concatenate((growth_array, grth.cpu().
                    numpy().flatten()))
                pred_array = np.concatenate((pred_array, pred_growth.cpu().
                    numpy().flatten()))
            mse /= len(val_loader.dataset)
            mae /= len(val_loader.dataset)
            r2 = r2_score(y_pred=pred_array, y_true=growth_array)
            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)
            print('\t\t%-6s \t MSE: %8.2f \t MAE: %8.2f \t R2: %+4.2f' % (
                val_loader.dataset.data_source, mse, mae, r2))
    return mse_list, mae_list, r2_list
