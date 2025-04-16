def imshow(img: torch.Tensor, savepath):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(savepath, dpi=400)
    plt.close()
