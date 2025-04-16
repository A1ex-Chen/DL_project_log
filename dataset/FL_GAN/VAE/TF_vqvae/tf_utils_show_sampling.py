def show_sampling(latent, recon, save_path, gray=False):
    plt.figure(figsize=(3, 2))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0, right=0.21, top=1, bottom=0.1,
        left=0.12)
    for i in range(len(recon)):
        plt.subplot(10, 2, 1 + i * 2)
        if gray:
            plt.imshow(latent[i], cmap='gray')
        else:
            plt.imshow(latent[i])
        plt.axis('off')
        plt.subplot(10, 2, 2 + i * 2)
        if gray:
            plt.imshow(recon[i], cmap='gray')
        else:
            plt.imshow(recon[i])
        plt.axis('off')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
