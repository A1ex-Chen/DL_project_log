def valid_drug_target(device: torch.device, drug_target_net: nn.Module,
    data_loader: torch.utils.data.DataLoader):
    drug_target_net.eval()
    correct_target = 0
    with torch.no_grad():
        for drug_feature, target in data_loader:
            drug_feature, target = drug_feature.to(device), target.to(device)
            out_target = drug_target_net(drug_feature)
            pred_target = out_target.max(1, keepdim=True)[1]
            correct_target += pred_target.eq(target.view_as(pred_target)).sum(
                ).item()
    target_acc = 100.0 * correct_target / len(data_loader.dataset)
    print('\tDrug Target Family Classification Accuracy: %5.2f%%' % target_acc)
    return target_acc
