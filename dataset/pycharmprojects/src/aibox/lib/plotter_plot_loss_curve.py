@staticmethod
def plot_loss_curve(global_batches: List[int],
    legend_to_losses_and_color_dict: Dict[str, Tuple[List[float], str]],
    path_to_plot: str):
    global_batches_length = len(global_batches)
    for losses, _ in legend_to_losses_and_color_dict.values():
        assert global_batches_length == len(losses)
    matplotlib.use('agg')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10.24, 7.68))
    for losses, color in legend_to_losses_and_color_dict.values():
        ax.plot(global_batches, losses, color)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    legends = list(legend_to_losses_and_color_dict.keys())
    ax.legend(legends, loc='upper left', bbox_to_anchor=(1, 1), shadow=True)
    ax.set_xlabel('Global Batch')
    ax.set_ylabel('Loss')
    ax.set_ylim(bottom=0)
    ax.set_title('Loss Curve')
    ax.grid()
    fig.savefig(path_to_plot)
    plt.close()
