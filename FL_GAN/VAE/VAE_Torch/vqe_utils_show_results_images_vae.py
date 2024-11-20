def show_results_images_vae(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = 'data'
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl')
            )
    train_losses, test_losses, samples, reconstructions, interpolations = fn(
        train_data, test_data)
    samples, reconstructions, interpolations = samples.astype('float32'
        ), reconstructions.astype('float32'), interpolations.astype('float32')
    print(
        f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, KL Loss: {test_losses[-1, 2]:.4f}'
        )
    plot_vae_training_plot(train_losses, test_losses,
        f'Dataset {dset_id} Train Plot')
    show_samples(samples, title=f'Dataset {dset_id} Samples')
    show_samples(reconstructions, title=f'Dataset {dset_id} Reconstructions')
    show_samples(interpolations, title=f'Dataset {dset_id} Interpolations')
