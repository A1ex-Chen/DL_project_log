def plot_metrics(history, title=None, skip_ep=0, outdir='.', add_lr=False):
    """Plots keras training curves history.
    Args:
        skip_ep: number of epochs to skip when plotting metrics
        add_lr: add curve of learning rate progression over epochs
    """

    def capitalize_metric(met):
        return ' '.join(s.capitalize() for s in met.split('_'))
    all_metrics = list(history.history.keys())
    pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m
        ]
    epochs = np.asarray(history.epoch) + 1
    if len(epochs) <= skip_ep:
        skip_ep = 0
    eps = epochs[skip_ep:]
    hh = history.history
    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m
        y_tr = hh[metric_name][skip_ep:]
        y_vl = hh[metric_name_val][skip_ep:]
        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim
        fig, ax1 = plt.subplots()
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth
            =1, alpha=0.6, label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--',
            linewidth=1, alpha=0.6, label=capitalize_metric(metric_name_val))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(capitalize_metric(metric_name))
        ax1.set_xlim([min(eps) - 1, max(eps) + 1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')
        if add_lr is True and 'lr' in hh:
            ax2 = ax1.twinx()
            ax2.plot(eps, hh['lr'][skip_ep:], color='g', marker='.',
                linestyle=':', linewidth=1, alpha=0.6, markersize=5, label='LR'
                )
            ax2.set_ylabel('Learning rate', color='g', fontsize=12)
            ax2.set_yscale('log')
            ax2.tick_params('y', colors='g')
        ax1.grid(True)
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None:
            plt.title(title)
        figpath = Path(outdir) / (metric_name + '.png')
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
