def plot_tune_results(csv_file='tune_results.csv'):
    """
    Plot the evolution results stored in an 'tune_results.csv' file. The function generates a scatter plot for each key
    in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on the plots.

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results. Defaults to 'tune_results.csv'.

    Examples:
        >>> plot_tune_results('path/to/tune_results.csv')
    """
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d

    def _save_one_file(file):
        """Save one matplotlib plot to 'file'."""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f'Saved {file}')
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)
    num_metrics_columns = 1
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    x = data.values
    fitness = x[:, 0]
    j = np.argmax(fitness)
    n = math.ceil(len(keys) ** 0.5)
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + num_metrics_columns]
        mu = v[j]
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap='viridis', alpha=0.8, edgecolors
            ='none')
        plt.plot(mu, fitness.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})
        plt.tick_params(axis='both', labelsize=8)
        if i % n != 0:
            plt.yticks([])
    _save_one_file(csv_file.with_name('tune_scatter_plots.png'))
    x = range(1, len(fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(x, fitness, marker='o', linestyle='none', label='fitness')
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ':', label='smoothed',
        linewidth=2)
    plt.title('Fitness vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    _save_one_file(csv_file.with_name('tune_fitness.png'))
