def on_epoch_end(self, epoch, logs=None):
    if args.dpsgd == True:
        print('\nDifferential Privacy Information')
        eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(args.
            dataset_len, args.batch_size, args.noise_multiplier, epoch + 1 +
            args.epochs, args.delta)
        logs['epsilon'] = eps
    checkpoint_path = (iwantto_path +
        f"/{args.dataset}/v1/{'dp' if args.dpsgd else 'normal'}/model")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path += f'/pixel_cnn_cp_{epoch}'
    self.model.save_weights(checkpoint_path)
