def valid_drug_qed(device: torch.device, drug_qed_net: nn.Module,
    data_loader: torch.utils.data.DataLoader):
    drug_qed_net.eval()
    mse, mae = 0.0, 0.0
    target_array, pred_array = np.array([]), np.array([])
    with torch.no_grad():
        for drug_feature, target in data_loader:
            drug_feature, target = drug_feature.to(device), target.to(device)
            pred_target = drug_qed_net(drug_feature)
            num_samples = target.shape[0]
            mse += F.mse_loss(pred_target, target).item() * num_samples
            mae += F.l1_loss(pred_target, target).item() * num_samples
            target_array = np.concatenate((target_array, target.cpu().numpy
                ().flatten()))
            pred_array = np.concatenate((pred_array, pred_target.cpu().
                numpy().flatten()))
        mse /= len(data_loader.dataset)
        mae /= len(data_loader.dataset)
        r2 = r2_score(y_pred=pred_array, y_true=target_array)
    print(
        '\tDrug Weighted QED Regression\n\t\tMSE: %8.6f \t MAE: %8.6f \t R2: %+4.2f'
         % (mse, mae, r2))
    return mse, mae, r2
