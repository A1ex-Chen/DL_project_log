def train_drug_qed(device: torch.device, drug_qed_net: nn.Module,
    data_loader: torch.utils.data.DataLoader, max_num_batches: int,
    loss_func: callable, optimizer: torch.optim):
    drug_qed_net.train()
    total_loss = 0.0
    num_samples = 0
    for batch_idx, (drug_feature, target) in enumerate(data_loader):
        if batch_idx >= max_num_batches:
            break
        drug_feature, target = drug_feature.to(device), target.to(device)
        drug_qed_net.zero_grad()
        pred_target = drug_qed_net(drug_feature)
        loss = loss_func(pred_target, target)
        loss.backward()
        optimizer.step()
        num_samples += target.shape[0]
        total_loss += loss.item() * target.shape[0]
    print('\tDrug Weighted QED Regression Loss: %8.6f' % (total_loss /
        num_samples))
