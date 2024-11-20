def train(net, trainloader, epochs, privacy_engine, device: str='cpu'):
    """Train the network on the training set."""
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-05
        )
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=net, optimizer=optimizer, data_loader=trainloader, epochs=
        epochs, target_epsilon=PARAMS['target_epsilon'], target_delta=
        PARAMS['target_delta'], max_grad_norm=PARAMS['max_grad_norm'])
    LOSS = 0.0
    train_fid = 0.0
    for e in tqdm.tqdm(range(epochs)):
        with BatchMemoryManager(data_loader=trainloader,
            max_physical_batch_size=PARAMS['max_batch_size'], optimizer=
            optimizer) as memory_safe_data_loader:
            loop = tqdm.tqdm(memory_safe_data_loader, total=len(trainloader
                ), leave=False)
            for images, labels in loop:
                optimizer.zero_grad()
                images = images.to(device)
                recon_images, mu, logvar = model(images)
                recon_loss = F.mse_loss(recon_images, images)
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) -
                    logvar.exp())
                loss = recon_loss + kld_loss
                LOSS += loss
                loss.backward()
                optimizer.step()
                loop.set_description(f'Epoch [{e}/{epochs}]')
                loop.set_postfix(loss=loss.item())
    epsilon, _ = optimizer.privacy_engine.get_privacy_spent(PRIVACY_PARAMS[
        'target_delta'])
    if images.shape[1] == 1:
        images = torch.repeat_interleave(images, repeats=3, dim=1)
        recon_images = torch.repeat_interleave(recon_images, repeats=3, dim=1)
    train_fid = compute_fid(images, recon_images, device)
    train_loss = LOSS.detach().item()
    results = {'train_loss': float(train_loss), 'fid': float(train_fid),
        'epsilon': epsilon}
    return results
