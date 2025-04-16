@TryExcept()
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(''), on_plot=None):
    """Plot training labels including class histograms and box statistics."""
    import pandas
    import seaborn
    warnings.filterwarnings('ignore', category=UserWarning, message=
        'The figure layout has changed to tight')
    warnings.filterwarnings('ignore', category=FutureWarning)
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)
    boxes = boxes[:1000000]
    x = pandas.DataFrame(boxes, columns=['x', 'y', 'width', 'height'])
    seaborn.pairplot(x, corner=True, diag_kind='auto', kind='hist',
        diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([(x / 255) for x in colors(i)])
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    seaborn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    seaborn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)
    boxes[:, 0:2] = 0.5
    boxes = ops.xywh2xyxy(boxes) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))
    ax[1].imshow(img)
    ax[1].axis('off')
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)
    fname = save_dir / 'labels.jpg'
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)
