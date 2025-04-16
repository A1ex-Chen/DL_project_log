def train_cl_clf(device: torch.device, category_clf_net: nn.Module,
    site_clf_net: nn.Module, type_clf_net: nn.Module, data_loader: torch.
    utils.data.DataLoader, max_num_batches: int, optimizer: torch.optim):
    category_clf_net.train()
    site_clf_net.train()
    type_clf_net.train()
    for batch_idx, (rnaseq, data_src, cl_site, cl_type, cl_category
        ) in enumerate(data_loader):
        if batch_idx >= max_num_batches:
            break
        rnaseq, data_src, cl_site, cl_type, cl_category = rnaseq.to(device
            ), data_src.to(device), cl_site.to(device), cl_type.to(device
            ), cl_category.to(device)
        category_clf_net.zero_grad()
        site_clf_net.zero_grad()
        type_clf_net.zero_grad()
        out_category = category_clf_net(rnaseq, data_src)
        out_site = site_clf_net(rnaseq, data_src)
        out_type = type_clf_net(rnaseq, data_src)
        F.nll_loss(input=out_category, target=cl_category).backward()
        F.nll_loss(input=out_site, target=cl_site).backward()
        F.nll_loss(input=out_type, target=cl_type).backward()
        optimizer.step()
