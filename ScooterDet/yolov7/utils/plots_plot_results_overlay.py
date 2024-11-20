def plot_results_overlay(start=0, stop=0):
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val',
        'val', 'Recall', 'mAP@0.5:0.95']
    t = ['Box', 'Objectness', 'Classification', 'P-R', 'mAP-F1']
    for f in sorted(glob.glob('results*.txt') + glob.glob(
        '../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11],
            ndmin=2).T
        n = results.shape[1]
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None
        fig.savefig(f.replace('.txt', '.png'), dpi=200)
