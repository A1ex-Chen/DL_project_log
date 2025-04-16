def show_latent(ori, latent, recon, save_path, gray=False):
    plt.figure(figsize=(3, 2))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0, right=0.25, top=0.94, bottom=
        0.1, left=0.12)
    for i in range(len(ori)):
        plt.subplot(10, 3, 1 + i * 3)
        if gray:
            plt.imshow(ori[i], cmap='gray')
        else:
            plt.imshow(ori[i])
        plt.axis('off')
        plt.subplot(10, 3, 2 + i * 3)
        if gray:
            plt.imshow(latent[i], cmap='gray')
        else:
            plt.imshow(latent[i])
        plt.axis('off')
        plt.subplot(10, 3, 3 + i * 3)
        if gray:
            plt.imshow(recon[i], cmap='gray')
        else:
            plt.imshow(recon[i])
        plt.axis('off')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
