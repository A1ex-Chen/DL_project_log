def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir=''):
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
        'val Box', 'val Objectness', 'val Classification', 'mAP@0.5',
        'mAP@0.5:0.95']
    if bucket:
        files = [('results%g.txt' % x) for x in id]
        c = ('gsutil cp ' + '%s ' * len(files) + '.') % tuple(
            'gs://%s/results%g.txt' % (bucket, x) for x in id)
        os.system(c)
    else:
        files = list(Path(save_dir).glob('results*.txt'))
    assert len(files
        ), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(
        save_dir)
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10,
                11], ndmin=2).T
            n = results.shape[1]
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan
                label = labels[fi] if len(labels) else f.stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2,
                    markersize=8)
                ax[i].set_title(s[i])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))
    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)
