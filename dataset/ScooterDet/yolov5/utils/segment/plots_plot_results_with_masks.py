def plot_results_with_masks(file='path/to/results.csv', dir='', best=True):
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files
        ), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f in files:
        try:
            data = pd.read_csv(f)
            index = np.argmax(0.9 * data.values[:, 8] + 0.1 * data.values[:,
                7] + 0.9 * data.values[:, 12] + 0.1 * data.values[:, 11])
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16,
                7, 8, 11, 12]):
                y = data.values[:, j]
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2,
                    markersize=2)
                if best:
                    ax[i].scatter(index, y[index], color='r', label=
                        f'best:{index}', marker='*', linewidth=3)
                    ax[i].set_title(s[j] + f'\n{round(y[index], 5)}')
                else:
                    ax[i].scatter(x[-1], y[-1], color='r', label='last',
                        marker='*', linewidth=3)
                    ax[i].set_title(s[j] + f'\n{round(y[-1], 5)}')
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()
