def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)
    criterion = nn.MSELoss()
    latent_loss_weight = 0.25
    sample_size = 25
    mse_sum = 0
    mse_n = 0
    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        img = img.to(device)
        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {'mse_sum': part_mse_sum, 'mse_n': part_mse_n}
        comm = dist.all_gather(comm)
        for part in comm:
            mse_sum += part['mse_sum']
            mse_n += part['mse_n']
        if dist.is_primary():
            lr = optimizer.param_groups[0]['lr']
            loader.set_description(
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; lr: {lr:.5f}'
                )
            if i % 100 == 0:
                model.eval()
                sample = img[:sample_size]
                with torch.no_grad():
                    out, _ = model(sample)
                utils.save_image(torch.cat([sample, out], 0),
                    f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                    nrow=sample_size, normalize=True, range=(-1, 1))
                model.train()
