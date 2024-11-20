def plot_error(y_true, y_pred, batch, file_ext, file_pre='output_dir',
    subsample=1000):
    if batch % 10:
        return
    total = len(y_true)
    if subsample and subsample < total:
        usecols = np.random.choice(total, size=subsample, replace=False)
        y_true = y_true[usecols]
        y_pred = y_pred[usecols]
    y_true = y_true * 100
    y_pred = y_pred * 100
    diffs = y_pred - y_true
    bins = np.linspace(-200, 200, 100)
    if batch == 0:
        y_shuf = np.random.permutation(y_true)
        plt.hist(y_shuf - y_true, bins, alpha=0.5, label='Random')
    plt.hist(diffs, bins, alpha=0.3, label='Epoch {}'.format(batch + 1))
    plt.title('Histogram of errors in percentage growth')
    plt.legend(loc='upper right')
    plt.savefig(file_pre + '.histogram' + file_ext + '.b' + str(batch) + '.png'
        )
    plt.close()
    fig, ax = plt.subplots()
    plt.grid('on')
    ax.scatter(y_true, y_pred, color='red', s=10)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(file_pre + '.diff' + file_ext + '.b' + str(batch) + '.png')
    plt.close()
