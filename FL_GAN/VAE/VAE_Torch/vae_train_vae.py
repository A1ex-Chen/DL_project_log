def train_vae(train_data, test_data):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """
    start_time = time.time()
    N, H, W, C = train_data.shape
    prior_p_z = dist.normal.Normal(0, 1)

    def dequantize(x, dequantize=True, reverse=False, alpha=0.1):
        with torch.no_grad():
            if reverse:
                return torch.ceil_(torch.sigmoid(x) * 255)
            if dequantize:
                x += torch.zeros_like(x).uniform_(0, 1)
            p = alpha / 2 + (1 - alpha) * x / 256
            return torch.log(p) - torch.log(1 - p)

    def loss_function(batch, output):

        def repeat(tensor, K=50):
            shape = (K,) + tuple(tensor.shape)
            return torch.cat(K * [tensor]).reshape(shape)
        K = 50
        x_mu, x_sigma, z, z_mu, z_sigma = output
        x_mu, x_sigma, z, z_mu, z_sigma = repeat(x_mu), repeat(x_sigma
            ), repeat(z), repeat(z_mu), repeat(z_sigma)
        k_batch = repeat(batch)
        log_p_z = prior_p_z.log_prob(z)
        z_normal = dist.normal.Normal(z_mu, z_sigma)
        posterior_log_prob = z_normal.log_prob(z)
        x_normal = dist.normal.Normal(x_mu, x_sigma)
        x_reconstruct_log_prob = x_normal.log_prob(k_batch)
        VLB = -torch.mean(log_p_z.sum(dim=(0, 2)) + x_reconstruct_log_prob.
            sum(dim=(0, 2, 3, 4)) - posterior_log_prob.sum(dim=(0, 2))) / K
        reconstruction_loss = -torch.mean(x_reconstruct_log_prob.sum(dim=(0,
            2, 3, 4))) / K
        KL = torch.mean(posterior_log_prob.sum(dim=(0, 2)) - log_p_z.sum(
            dim=(0, 2))) / K
        return VLB, reconstruction_loss, KL
    batch_size = 128
    dataset_params = {'batch_size': batch_size, 'shuffle': True}
    print('[INFO] Creating model and data loaders')
    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])
        ).float().cuda()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float(
        ).cuda()
    train_loader = torch.utils.data.DataLoader(dequantize(train_data), **
        dataset_params)
    test_loader = torch.utils.data.DataLoader(dequantize(test_data), **
        dataset_params)
    n_epochs = 20 if dset_id == 1 else 50
    lr = 0.001 if dset_id == 1 else 0.0005
    latent_dim = 16 if dset_id == 1 else 32
    vae = VAE(latent_dim=latent_dim).cuda()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    train_losses = []
    init_test_loss = get_batched_loss(test_loader, vae, loss_function)
    test_losses = [[*init_test_loss]]
    print(
        f'Initial loss-variables: VLB: {test_losses[0][0]:.2f}, Reconstruct loss: {test_losses[0][1]:.2f}, KL: {test_losses[0][2]:.2f}'
        )
    print('[INFO] Training')
    for epoch in range(n_epochs):
        epoch_start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            output = vae(batch)
            loss = loss_function(batch, output)
            loss[0].backward()
            optimizer.step()
            train_losses.append([loss[0].cpu().item(), loss[1].cpu().item(),
                loss[2].cpu().item()])
        test_loss = get_batched_loss(test_loader, vae, loss_function)
        test_losses.append([*test_loss])
        print(
            f'[{100 * (epoch + 1) / n_epochs:.2f}%] Epoch {epoch + 1} - Test loss (VLB/RL/KL): {test_loss[0]:.2f}/{test_loss[1]:.2f}/{test_loss[2]:.2f} - Time elapsed: {time.time() - epoch_start:.2f}'
            )

    def sample(noise=True, no_samples=100):
        """
        Sample z ~ p(z) and x ~ p(x|z) if noise, x = mu(z) otherwise
        """
        sample_shape = no_samples, latent_dim
        z = prior_p_z.sample(sample_shape).cuda()
        x_mu = vae.decoder(z)
        x_sigma = torch.ones_like(x_mu)
        x = vae.reparameterize(x_mu, x_sigma, noise)
        samples = dequantize(x, reverse=True).detach().cpu().numpy()
        return np.transpose(samples, [0, 2, 3, 1])

    def reconstruction_pairs(no_reconstructions=50):
        """
        Creating reconstruction pairs (x, x') where x is the original image and x' is the decoder-output
        """
        x_original = test_data[:no_reconstructions]
        x_dequantized = dequantize(x_original, dequantize=False)
        x_reconstructed = vae(x_dequantized)[0]
        x_reconstructed = dequantize(x_reconstructed, reverse=True)
        pairs = torch.zeros_like(torch.cat((x_original, x_reconstructed),
            dim=0)).detach().cpu().numpy()
        pairs[::2] = x_original.detach().cpu().numpy()
        pairs[1::2] = x_reconstructed.detach().cpu().numpy()
        pairs = np.clip(pairs, 0, 255)
        return np.transpose(pairs, [0, 2, 3, 1])

    def interpolate_images(no_interpolations=10):
        interpolations = torch.zeros(size=(no_interpolations * 10, C, H, W)
            ).float().cuda()
        counter = 0
        weights = np.linspace(0, 1, 10)[1:-1]
        for i in range(no_interpolations):
            x_a, x_b = test_data[i].unsqueeze(0), test_data[i +
                no_interpolations].unsqueeze(0)
            x_a_dequantized, x_b_dequantized = dequantize(x_a, dequantize=False
                ), dequantize(x_b, dequantize=False)
            z_a, z_b = vae(x_a_dequantized)[2], vae(x_b_dequantized)[2]
            interpolations[counter] = x_a
            counter += 1
            for weight in weights:
                z_interpolated = (1 - weight) * z_a + weight * z_b
                x_interpolated = vae.decoder(z_interpolated)
                x_interpolated = dequantize(x_interpolated, reverse=True)
                interpolations[counter] = x_interpolated[0]
                counter += 1
            interpolations[counter] = x_b
            counter += 1
        interpolations = interpolations.detach().cpu().numpy()
        interpolations = np.clip(interpolations, 0, 255)
        return np.transpose(interpolations, [0, 2, 3, 1])
    torch.cuda.empty_cache()
    vae.eval()
    with torch.no_grad():
        print('[INFO] Sampling images')
        samples = sample(noise=False)
        print('[INFO] Creating reconstructing pairs')
        pairs = reconstruction_pairs()
        print('[INFO] Interpolating')
        interpolations = interpolate_images()
    print(f'[DONE] Time elapsed: {time.time() - start_time:.2f} s')
    train_losses, test_losses = np.array(train_losses), np.array(test_losses)
    print('Samples', samples.shape)
    print('Pairs', pairs.shape)
    print('Interpolations', interpolations.shape)
    return np.array(train_losses), np.array(test_losses
        ), samples, pairs, interpolations
