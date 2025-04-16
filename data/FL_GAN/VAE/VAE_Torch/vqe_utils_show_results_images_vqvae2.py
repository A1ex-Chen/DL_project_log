def show_results_images_vqvae2(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = 'data'
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl')
            )
    (vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses,
        pixelcnn_test_losses, samples, reconstructions) = fn(train_data,
        test_data, dset_id)
    samples, reconstructions = samples.astype('float32'
        ), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    show_training_plot(vqvae_train_losses, vqvae_test_losses,
        f'Dataset {dset_id} VQ-VAE Train Plot')
    show_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,
        f'Dataset {dset_id} PixelCNN Prior Train Plot')
    show_samples(samples, title=f'Dataset {dset_id} Samples')
    show_samples(reconstructions, title=f'Dataset {dset_id} Reconstructions')
