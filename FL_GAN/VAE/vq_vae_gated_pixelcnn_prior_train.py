def train(args, data_shape, model, optimizer, no_epochs, train_loader,
    validation_loader, test_loader, privacy_engine, prior_only=False):
    """
    Trains model and returns training and test losses
    """
    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    device = args.device
    model.to(device)
    train_losses = []
    test_losses = []
    validate_losses = []
    if not prior_only:
        model_name = 'VQ-VAE'
        loss_fct = get_vae_loss
        recon_images, recon_labels = next(iter(test_loader))
        recon_images = recon_images[:args.num_reconstruction]
        recon_labels = recon_labels[:args.num_reconstruction]
        DATASET = args.dataset
        DP = args.dp
        CLIENT = args.client
        label_names = get_labels(DATASET)
        val_labels_name = [label_names[i] for i in np.array(recon_labels)]
        image_save_path = f'Results/{DATASET}/{CLIENT}/{DP}/G_data'
        label_save_path = f'Results/{DATASET}/{CLIENT}/{DP}/Labels'
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        if not os.path.exists(label_save_path):
            os.makedirs(label_save_path)
        image_save_path += f'/grand_truth_images.png'
        label_save_path += f'/validation_labels.txt'
        np.savetxt(label_save_path, val_labels_name, fmt='%s')
        imshow(torchvision.utils.make_grid(recon_images, nrow=6),
            image_save_path)
    else:
        model_name = 'Gated Pixel-CNN'
        loss_fct = get_pixelcnn_prior_loss
    print(f'[INFO] Training {model_name} on Device {device}')
    if args.dp == 'gaussian':
        EPSILON = []
    min_loss = math.inf
    for epoch in tqdm.tqdm(range(no_epochs)):
        model.train()
        epoch_start = time.time()
        loop = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)
        print(f'\n[INFO] Training ...')
        for images, labels in loop:
            optimizer.zero_grad()
            images = images.to(device)
            output = model(images)
            loss = loss_fct(images, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.cpu().item())
            loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(loss=loss.item())
        print(f'\n[INFO] Validation ...')
        validate_loss = get_batched_loss(args, validation_loader, model,
            loss_fct, loss_triples=False)
        validate_losses.append(validate_loss)
        print(f'\n[INFO] Test ...')
        test_loss = get_batched_loss(args, test_loader, model, loss_fct,
            loss_triples=False)
        test_losses.append(test_loss)
        if not prior_only:
            reconstruction_save_path = (
                f'Results/{DATASET}/{CLIENT}/{DP}/G_data/Reconstruction')
            if not os.path.exists(reconstruction_save_path):
                os.makedirs(reconstruction_save_path)
            recon_image = model(recon_images.to(device))[0]
            recon_images_save_path = (reconstruction_save_path +
                f'/reconstruction_images_at_epoch_{epoch + 1:03d}_G.png')
            imshow(torchvision.utils.make_grid(recon_image.cpu().detach(),
                nrow=6), recon_images_save_path)
        else:
            sampling_save_path = (
                f'Results/{DATASET}/{CLIENT}/{DP}/G_data/Sampling')
            if not os.path.exists(sampling_save_path):
                os.makedirs(sampling_save_path)
            N, H, W, C = data_shape

            def sample(no_samples=args.num_sampling):
                shape = no_samples, H // 4, W // 4
                q_samples = torch.zeros(size=shape).long().to(device)
                for i in range(H // 4):
                    for j in range(W // 4):
                        out = model.pixelcnn_prior(q_samples)
                        proba = F.softmax(out, dim=1)
                        q_samples[:, i, j] = torch.multinomial(proba[:, :,
                            i, j], 1).squeeze().float()
                latents_shape = q_samples.shape
                encoding_inds = q_samples.view(-1, 1)
                encoding_one_hot = torch.zeros(encoding_inds.size(0), args.
                    num_embeddings, device=device)
                encoding_one_hot.scatter_(1, encoding_inds, 1)
                quantized_latents = torch.matmul(encoding_one_hot, model.
                    codebook.codebook.weight)
                quantized_latents = quantized_latents.view(latents_shape +
                    (args.embedding_dim,))
                z_q_samples = quantized_latents.permute(0, 3, 1, 2).contiguous(
                    )
                x_samples = model.decoder(z_q_samples)
                return x_samples.detach().cpu()
            sampling_images = sample(args.num_sampling)
            sample_images_save_path = (sampling_save_path +
                f'/sampling_images_at_epoch_{epoch + 1:03d}.png')
            imshow(torchvision.utils.make_grid(sampling_images, nrow=6),
                sample_images_save_path)
            print(
                f'[{100 * (epoch + 1) / no_epochs:.2f}%] Epoch {epoch + 1} - Train loss: {np.mean(train_losses):.2f} - Validate loss: {validate_loss:.2f} - Test loss: {test_loss:.2f} - Time elapsed: {time.time() - epoch_start:.2f}'
                , end='')
        if args.dp == 'gaussian':
            epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
            EPSILON.append(epsilon.item())
            print(f'(ε = {epsilon:.2f}, δ = {args.delta})')
        else:
            print('\n')
        if prior_only:
            if test_loss < min_loss:
                min_loss = test_loss
                weight_save_pth = f'Results/{DATASET}/{CLIENT}/{DP}/weights'
                if not os.path.exists(weight_save_pth):
                    os.makedirs(weight_save_pth)
                weight_save_pth += f'/weights_central_{args.epochs}.pt'
                torch.save(model.state_dict(), weight_save_pth)
        if not prior_only:
            metrics = {'vq_vae_train_losses': np.array(train_losses),
                'vq_vae_validate_losses': np.array(validate_losses),
                'vq_vae_test_losses': np.array(test_losses)}
        else:
            metrics = {'pixcel_cnn_train_losses': np.array(train_losses),
                'pixcel_cnn_validate_losses': np.array(validate_losses),
                'pixcel_cnn_test_losses': np.array(test_losses)}
        if DP == 'gaussian':
            metrics['epsilon'] = np.array(EPSILON)
        return metrics
