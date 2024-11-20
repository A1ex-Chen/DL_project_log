def latent_distribution(mu, labels, info):
    e = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(mu
        .detach().cpu())
    plt.figure()
    plt.scatter(e[:, 0], e[:, 1], c=labels, cmap='tab10')
    plt.colorbar(ticks=np.arange(10), boundaries=np.arange(11) - 0.5)
    latent_save_path = (
        f"Results/{info['dataset']}/{info['client']}/{info['dp']}/Figures/Latent"
        )
    if not os.path.exists(latent_save_path):
        os.makedirs(latent_save_path)
    latent_save_path += f"/latent_distribution_{info['current_epoch']}.png"
    plt.savefig(latent_save_path, dpi=400)
    plt.close()
