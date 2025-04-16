def main():
    """
    ## Load dataset
    """
    train_data, train_labels, test_data, test_labels = load_data(args.dataset)
    data_variance = np.var(train_data, dtype=np.float32)
    """
    ## Train the VAE model
    """
    data_shape = train_data.shape[1:]
    vae_trainer = VAETrainer(data_variance, args.dataset, data_shape=data_shape
        )
    print(f'Differential Privacy Switch: {args.dpsgd}')
    if args.dpsgd:
        print('Processing in Differential Privacy')
        args.dataset_len = train_data.shape[0]
        magnitude = len(str(args.dataset_len))
        args.delta = pow(10, -magnitude)
        optimizer = VectorizedDPAdam(l2_norm_clip=args.l2_norm_clip,
            noise_multiplier=args.noise_multiplier, num_microbatches=args.
            micro_batch, learning_rate=args.learning_rate)
    else:
        print('Processing in normal')
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.
            learning_rate)
    vae_trainer.compile(optimizer=optimizer)
    print(
        f"Start to training with {'DP' if args.dpsgd else 'Normal'} for {args.dataset}"
        )
    train_start = time.time()
    history = vae_trainer.fit(train_data, epochs=args.epochs, batch_size=
        args.batch_size, callbacks=[CustomCallback()])
    train_end = time.time() - train_start
    print(f'VQ-VAE training time: {train_end:.2f}s')
    vqvae_metric_save_path = (iwantto_path +
        f"/{args.dataset}/vae/{'dp' if args.dpsgd else 'normal'}/metric")
    if not os.path.exists(vqvae_metric_save_path):
        os.makedirs(vqvae_metric_save_path)
    vqvae_metric_save = (vqvae_metric_save_path +
        f'/vq_vae_metrics_{args.epochs}.csv')
    vq_vae_metrics = pd.DataFrame(history.history)
    vq_vae_metrics.to_csv(vqvae_metric_save, index=False)
    """
    Reconstruction results on the test set
    Save True image and generated image
    VAE
    """
    reconstruction_path_traditional_flow = (iwantto_path +
        f"/{args.dataset}/vae/{'dp' if args.dpsgd else 'normal'}/Images")
    if not os.path.exists(reconstruction_path_traditional_flow):
        os.makedirs(reconstruction_path_traditional_flow)
    truth_path = (reconstruction_path_traditional_flow +
        f'/grand_truth_images_{args.epochs}.png')
    trained_vqvae_model = vae_trainer.vqvae
    idx = args.recon_num
    test_images = test_data[:idx]
    label_save_path = (reconstruction_path_traditional_flow +
        '/grand_truth_label.txt')
    label_names = get_labels(args.dataset)
    val_labels_name = [label_names[i] for i in np.array(test_labels[:idx])]
    np.savetxt(label_save_path, val_labels_name, fmt='%s')
    batch_size = int(pow(args.recon_num, 0.5))
    reconstruction_images = trained_vqvae_model.predict(test_images)
    reconstruction_save_path = (reconstruction_path_traditional_flow +
        f'/reconstruction_image_{args.epochs}.png')
    info_f = open(vqvae_metric_save_path + f'/info_{args.epochs}.txt', 'w')
    fid = get_fid_score(tf.cast(test_images * 255, tf.int32), tf.cast(
        reconstruction_images * 255, tf.int32))
    print(f'Compare test images with reconstruction images, FID: {fid:.2f}')
    info_f.write(
        f'Compare test images with reconstruction images, FID: {fid:.2f}\n')
    inception_score = get_inception_score(tf.cast(reconstruction_images * 
        255, tf.int32))
    print(f'Compute reconstruction images, IS: {inception_score:.2f}')
    info_f.write(f'Compute reconstruction images, IS: {inception_score:.2f}\n')
    psnr = get_psnr(tf.cast(test_images * 255, tf.int32), tf.cast(
        reconstruction_images * 255, tf.int32))
    print(f'Peak Signal-to-Noise Ratio, PSNR: {psnr:.2f}')
    info_f.write(f'Peak Signal-to-Noise Ratio, PSNR: {psnr:.2f}\n')
    info_f.write(f'VAE training time: {train_end:.2f}s\n')
    show_batch(test_images, batch_size, truth_path, False if test_images.
        shape[3] == 3 else True)
    show_batch(tf.cast(reconstruction_images * 255, tf.int32), batch_size,
        reconstruction_save_path, False if reconstruction_images.shape[3] ==
        3 else True)
    """
    ## Sampling
    """
    decoder = vae_trainer.vqvae.get_layer('decoder')
    generated_samples = decoder.predict(quantized)
    sampling_save_path = (reconstruction_path_traditional_flow +
        f'/sampling_image_{args.epochs}.png')
    show_sampling(priors, tf.cast(generated_samples * 255, tf.int32),
        sampling_save_path, False if generated_samples.shape[3] == 3 else True)
    inception_score = get_inception_score(tf.cast(generated_samples * 255,
        tf.int8))
    print(f'Compute sampling images, IS: {inception_score:.2f}')
    info_f.write(f'Compute sampling images, IS: {inception_score:.2f}\n')
