def imshow_cls(im, labels=None, pred=None, names=None, nmax=25, verbose=
    False, f=Path('images.jpg')):
    from utils.augmentations import denormalize
    names = names or [f'class{i}' for i in range(1000)]
    blocks = torch.chunk(denormalize(im.clone()).cpu().float(), len(im), dim=0)
    n = min(len(blocks), nmax)
    m = min(8, round(n ** 0.5))
    fig, ax = plt.subplots(math.ceil(n / m), m)
    ax = ax.ravel() if m > 1 else [ax]
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(
            0.0, 1.0))
        ax[i].axis('off')
        if labels is not None:
            s = names[labels[i]] + (f'—{names[pred[i]]}' if pred is not
                None else '')
            ax[i].set_title(s, fontsize=8, verticalalignment='top')
    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        LOGGER.info(f'Saving {f}')
        if labels is not None:
            LOGGER.info('True:     ' + ' '.join(f'{names[i]:3s}' for i in
                labels[:nmax]))
        if pred is not None:
            LOGGER.info('Predicted:' + ' '.join(f'{names[i]:3s}' for i in
                pred[:nmax]))
    return f
