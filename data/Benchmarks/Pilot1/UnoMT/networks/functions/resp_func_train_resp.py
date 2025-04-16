def train_resp(device: torch.device, resp_net: nn.Module, data_loader:
    torch.utils.data.DataLoader, max_num_batches: int, loss_func: callable,
    optimizer: torch.optim):
    resp_net.train()
    total_loss = 0.0
    num_samples = 0
    for batch_idx, (rnaseq, drug_feature, conc, grth) in enumerate(data_loader
        ):
        if batch_idx >= max_num_batches:
            break
        rnaseq, drug_feature, conc, grth = rnaseq.to(device), drug_feature.to(
            device), conc.to(device), grth.to(device)
        resp_net.zero_grad()
        pred_growth = resp_net(rnaseq, drug_feature, conc)
        loss = loss_func(pred_growth, grth)
        loss.backward()
        optimizer.step()
        num_samples += conc.shape[0]
        total_loss += loss.item() * conc.shape[0]
    print('\tDrug Response Regression Loss: %8.2f' % (total_loss / num_samples)
        )
