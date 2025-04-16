def plot(plotting_recall_array_: np.ndarray, plotting_precision_array_: np.
    ndarray, plotting_f1_score_array_: np.ndarray, plotting_prob_array_: np
    .ndarray, category: str, ap: float, path_to_plot: str):
    matplotlib.use('agg')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12.8, 7.68))
    bar_width = 0.2
    pos = np.arange(plotting_prob_array_.shape[0])
    ax.bar(pos - bar_width, plotting_precision_array_, bar_width, color=
        'blue', edgecolor='none')
    ax.bar(pos, plotting_recall_array_, bar_width, color='green', edgecolor
        ='none')
    ax.bar(pos + bar_width, plotting_f1_score_array_, bar_width, color=
        'purple', edgecolor='none')
    for i, p in enumerate(pos):
        ax.text(p - bar_width, plotting_precision_array_[i] + 0.002,
            f'{plotting_precision_array_[i]:.3f}', color='blue', fontsize=6,
            rotation=90, ha='center', va='bottom')
        ax.text(p + 0.05, plotting_recall_array_[i] + 0.002,
            f'{plotting_recall_array_[i]:.3f}', color='green', fontsize=6,
            rotation=90, ha='center', va='bottom')
        ax.text(p + bar_width + 0.05, plotting_f1_score_array_[i] + 0.002,
            f'{plotting_f1_score_array_[i]:.3f}', color='purple', fontsize=
            6, rotation=90, ha='center', va='bottom')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    legends = ['Precision', 'Recall', 'F1-Score']
    ax.legend(legends, loc='upper left', bbox_to_anchor=(1, 1), shadow=True)
    ax.set_xlabel('Confidence Threshold')
    plt.xticks(pos, [f'{it:.4f}' for it in plotting_prob_array_], rotation=45)
    ax.set_ylim([0.0, 1.1])
    ax.set_title(f'Threshold versus PR: {category} AP = {ap:.4f}')
    ax.grid()
    fig.savefig(path_to_plot)
    plt.close()
