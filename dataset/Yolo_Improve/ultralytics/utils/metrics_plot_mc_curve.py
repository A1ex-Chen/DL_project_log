@plt_settings()
def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel=
    'Confidence', ylabel='Metric', on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')
    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=
        f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)
