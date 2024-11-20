def train_drug_target(device: torch.device, drug_target_net: nn.Module,
    data_loader: torch.utils.data.DataLoader, max_num_batches: int,
    optimizer: torch.optim):
    drug_target_net.train()
    for batch_idx, (drug_feature, target) in enumerate(data_loader):
        if batch_idx >= max_num_batches:
            break
        drug_feature, target = drug_feature.to(device), target.to(device)
        drug_target_net.zero_grad()
        out_target = drug_target_net(drug_feature)
        F.nll_loss(input=out_target, target=target).backward()
        optimizer.step()
