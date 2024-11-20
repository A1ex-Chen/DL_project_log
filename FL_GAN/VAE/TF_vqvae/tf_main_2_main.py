def main():
    """
    ## Load dataset
    """
    train_data, train_labels, test_data, test_labels = load_data(args.dataset)
    data_variance = np.var(train_data, dtype=np.float32)
    """
    ## Train the VQ-VAE model
    """
    data_shape = train_data.shape[1:]
    vqvae_trainer = VQVAETrainer(data_variance, embedding_dim=args.
        embedding_dim, num_embeddings=args.num_embeddings, data_shape=
        data_shape)
    print(f'V2: Differential Privacy Switch: {args.dpsgd}')
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
    vqvae_trainer.compile(optimizer=optimizer, run_eagerly=True)
    train_start = time.time()
    print(f"Start to training with {'DP' if args.dpsgd else 'Normal'}")
    history = vqvae_trainer.fit(train_data, epochs=args.epochs, batch_size=
        args.batch_size, callbacks=[CustomCallback()], verbose=1)
    train_end = time.time() - train_start - history.history['Extra time'][0]
    print(f'VQ-VAE and Pixel CNN training time: {train_end:.2f}s')
    vqvae_metric_save_path = (iwantto_path +
        f"/{args.dataset}/v2/{'dp' if args.dpsgd else 'normal'}/metric")
    if not os.path.exists(vqvae_metric_save_path):
        os.makedirs(vqvae_metric_save_path)
    vqvae_metric_path = (vqvae_metric_save_path +
        f'/vq_vae_metrics_{args.epochs}.csv')
    vq_vae_metrics = pd.DataFrame(history.history)
    vq_vae_metrics.to_csv(vqvae_metric_path, index=False)
    """
    Reconstruction results on the test set
    Save True image and generated image
    This is own flow: v2
    """
    reconstruction_path_traditional_flow = (iwantto_path +
        f"/{args.dataset}/v2/{'dp' if args.dpsgd else 'normal'}/Images")
    if not os.path.exists(reconstruction_path_traditional_flow):
        os.makedirs(reconstruction_path_traditional_flow)
    truth_path = (reconstruction_path_traditional_flow +
        f'/grand_truth_images_{args.epochs}.png')
    trained_vqvae_model = vqvae_trainer.vqvae
    idx = args.recon_num
    test_images = test_data[:idx]
    label_save_path = (reconstruction_path_traditional_flow +
        '/grand_truth_label.txt')
    label_names = get_labels(args.dataset)
    val_labels_name = [label_names[i] for i in np.array(test_labels[:idx])]
    np.savetxt(label_save_path, val_labels_name, fmt='%s')
    batch_size = int(pow(args.recon_num, 0.5))
    reconstruction_image = trained_vqvae_model.predict(test_images)
    reconstruction_save_path = (reconstruction_path_traditional_flow +
        f'/reconstruction_image_{args.epochs}.png')
    info_f = open(vqvae_metric_save_path + f'/info_{args.epochs}.txt', 'w')
    fid = get_fid_score(tf.cast(test_images * 255, tf.int32), tf.cast(
        reconstruction_image * 255, tf.int32))
    print(f'Compare test images with reconstruction images, FID: {fid:.2f}')
    info_f.write(
        f'Compare test images with reconstruction images, FID: {fid:.2f}\n')
    inception_score = get_inception_score(tf.cast(reconstruction_image * 
        255, tf.int32))
    print(f'Compute reconstruction images, IS: {inception_score:.2f}')
    info_f.write(f'Compute reconstruction images, IS: {inception_score:.2f}\n')
    psnr = get_psnr(tf.cast(test_images * 255, tf.int32), tf.cast(
        reconstruction_image * 255, tf.int32))
    print(f'Peak Signal-to-Noise Ratio, PSNR: {psnr:.2f}')
    info_f.write(f'Peak Signal-to-Noise Ratio, PSNR: {psnr:.2f}\n')
    info_f.write(f'VQ-VAE and Pixel CNN training time: {train_end:.2f}s\n')
    show_batch(test_images, batch_size, truth_path, False if test_images.
        shape[3] == 3 else True)
    show_batch(tf.cast(reconstruction_image * 255, tf.int32), batch_size,
        reconstruction_save_path, False if reconstruction_image.shape[3] ==
        3 else True)
    """
    ## Visualizing the discrete codes
    """
    encoder = vqvae_trainer.vqvae.get_layer('encoder')
    quantizer = vqvae_trainer.vqvae.get_layer('vector_quantizer')
    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.
        shape[:-1])
    latent_save_path = (reconstruction_path_traditional_flow +
        f'/latent_image_{args.epochs}.png')
    show_latent(test_images[:args.latent_num], codebook_indices[:args.
        latent_num], tf.cast(reconstruction_image[:args.latent_num] * 255,
        tf.int32), latent_save_path)
    pixel_cnn = vqvae_trainer.get_layer('pixel_cnn')
    pixel_cnn.compile(optimizer=optimizer, loss=keras.losses.
        SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    ar_acc = []
    for _ in range(args.test_num):
        idx = np.random.choice(len(test_data), int(len(test_data) * 0.8))
        test_images = test_data[idx]
        encoded_outputs = encoder.predict(test_images)
        flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.
            shape[-1])
        codebook_indices = np.array([])
        for i in tqdm.tqdm(range(int(flat_enc_outputs.shape[0] / 1000))):
            if i == int(flat_enc_outputs.shape[0] / 1000) - 1:
                codebook_indices = np.concatenate((codebook_indices,
                    quantizer.get_code_indices(flat_enc_outputs[i * 1000:])
                    .numpy()), axis=0)
            else:
                codebook_indices = np.concatenate((codebook_indices,
                    quantizer.get_code_indices(flat_enc_outputs[i * 1000:(i +
                    1) * 1000]).numpy()), axis=0)
        codebook_indices = codebook_indices.reshape(encoded_outputs.shape[:-1])
        _, acc = pixel_cnn.evaluate(codebook_indices, codebook_indices,
            batch_size=args.batch_size, verbose=0)
        ar_acc.append(acc)
    pixel_cnn_save_path = (iwantto_path +
        f"/{args.dataset}/v2/{'dp' if args.dpsgd else 'normal'}/metric")
    if not os.path.exists(pixel_cnn_save_path):
        os.makedirs(pixel_cnn_save_path)
    pixel_cnn_acc_save_path = (pixel_cnn_save_path +
        f'/pixel_cnn_acc_{args.epochs}.csv')
    pd.DataFrame(ar_acc).to_csv(pixel_cnn_acc_save_path, index=False)
    """
    ## Codebook sampling
    """
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    outputs = pixel_cnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.
        Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)
    batch = args.sampling_num
    priors = np.zeros(shape=(batch,) + pixel_cnn.input_shape[1:])
    batch, rows, cols = priors.shape
    for row in range(rows):
        for col in range(cols):
            probs = sampler.predict(priors, verbose=0)
            priors[:, row, col] = probs[:, row, col]
    print(f'Prior shape: {priors.shape}')
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype('int32'), vqvae_trainer.
        num_embeddings).numpy()
    quantized = tf.matmul(priors_ohe.astype('float32'),
        pretrained_embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *encoded_outputs.shape[1:]))
    decoder = vqvae_trainer.vqvae.get_layer('decoder')
    generated_samples = decoder.predict(quantized)
    sampling_save_path = (reconstruction_path_traditional_flow +
        f'/sampling_image_{args.epochs}.png')
    show_sampling(priors, tf.cast(generated_samples * 255, tf.int32),
        sampling_save_path, False if generated_samples.shape[3] == 3 else True)
    inception_score = get_inception_score(tf.cast(generated_samples * 255,
        tf.int8))
    print(f'Compute sampling images, IS: {inception_score:.2f}')
    info_f.write(f'Compute sampling images, IS: {inception_score:.2f}\n')
    info_f.close()
