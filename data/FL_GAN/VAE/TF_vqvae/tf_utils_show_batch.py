def show_batch(image_batch, batch_size, save_path, gray=False):
    plt.figure(figsize=(5, 5))
    for n in range(batch_size * batch_size):
        plt.subplot(batch_size, batch_size, n + 1)
        if gray:
            plt.imshow(image_batch[n], cmap='gray')
        else:
            plt.imshow(image_batch[n])
        plt.axis('off')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
