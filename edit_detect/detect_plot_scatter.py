def plot_scatter(x_set_list: List[List[torch.Tensor]], y_set_list: List[
    List[torch.Tensor]], color_set_list: List[str]=['tab:blue',
    'tab:orange', 'tab:red'], zorder_set_list: List[str]=[0, 2, 1],
    label_set_list: List[str]=['Training Real', 'Out Dist Real', 'Fake'],
    title: str=f'Flow-In Rate at Timestep', fig_name: str='scatter.jpg',
    xlabel: str='Mean', ylabel: str='Variance', xscale: str='linear',
    yscale: str='linear'):
    min_len: int = min(len(x_set_list), len(y_set_list), len(color_set_list
        ), len(label_set_list), len(zorder_set_list))
    fig, ax = plt.subplots(figsize=(8, 5))
    for x, y, color, label, zorder in zip(x_set_list[:min_len], y_set_list[
        :min_len], color_set_list[:min_len], label_set_list[:min_len],
        zorder_set_list[:min_len]):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()
        ax.scatter(x, y, c=color, label=label, alpha=0.8, edgecolors='none',
            zorder=zorder)
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    plt.savefig(fig_name)
    plt.show()
