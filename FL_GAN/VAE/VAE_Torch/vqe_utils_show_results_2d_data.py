def show_results_2d_data(dset_id, fn):
    train_data, test_data = sample_2d_data(dset_id)
    train_losses, test_losses, samples_noise, samples_nonoise = fn(train_data,
        test_data)
    print(
        f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, KL Loss: {test_losses[-1, 2]:.4f}'
        )
    plot_vae_training_plot(train_losses, test_losses,
        f'Dataset {dset_id} Train Plot')
    save_scatter_2d(samples_noise, title='Samples with Decoder Noise')
    save_scatter_2d(samples_nonoise, title='Samples without Decoder Noise')
