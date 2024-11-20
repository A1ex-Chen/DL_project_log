def plot_results(self, models_to_plot=None):
    """
        Plot the binning proportions of different methods
        :param models_to_plot: optional list of model labels to plot
        """
    K = self.number_of_bins
    w = 1.0 / (len(self.cached_results) + 1)
    assert K == self.bin_proportions.size
    assert self.cached_results

    def calc_se(p1, n1, p2, n2):
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        return np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if not models_to_plot:
        models_to_plot = sorted(list(self.cached_results.keys()))
    train_se = calc_se(self.bin_proportions, self.ref_sample_size, self.
        bin_proportions, self.cached_results[models_to_plot[0]]['N'])
    plt.bar(np.arange(0, K) + 0.5, height=train_se * 2.0, bottom=self.
        bin_proportions - train_se, width=1.0, label='Train$\\pm$SE', color
        ='gray')
    ymax = 0.0
    for i, model in enumerate(models_to_plot):
        results = self.cached_results[model]
        label = '%s (%i : %.4f)' % (model, results['NDB'], results['JS'])
        ymax = max(ymax, np.max(results['Proportions']))
        if K <= 70:
            plt.bar(np.arange(0, K) + (i + 1.0) * w, results['Proportions'],
                width=w, label=label)
        else:
            plt.plot(np.arange(0, K) + 0.5, results['Proportions'], '--*',
                label=label)
    plt.legend(loc='best')
    plt.ylim((0.0, min(ymax, np.max(self.bin_proportions) * 4.0)))
    plt.grid(True)
    plt.title('Binning Proportions Evaluation Results for {} bins (NDB : JS)'
        .format(K))
    plt.show()
