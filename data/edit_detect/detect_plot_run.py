def plot_run(x_set_list: List[List[torch.Tensor]], color_set_list: List[str
    ]=['tab:blue', 'tab:orange', 'tab:red'], label_set_list: List[str]=[
    'Training Real', 'Out Dist Real', 'Fake'], title: str=
    f'Reconst Loss at Timestep', fig_name: str='line.jpg', is_plot_var:
    bool=True):
    min_len: int = min(len(x_set_list), len(color_set_list), len(
        label_set_list))
    fig, ax = plt.subplots(figsize=(8, 5))
    for x, color, label in zip(x_set_list[:min_len], color_set_list[:
        min_len], label_set_list[:min_len]):
        print(f'x: {type(x)}')
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        else:
            x = torch.tensor(x)
        print(f'x: {len(x)}')
        for i, line in enumerate(x):
            if len(line.shape) == 2:
                if is_plot_var:
                    line_mean: torch.Tensor = line.mean(0)
                    line_var: torch.Tensor = line.var(0)
                    if i == 0:
                        print(
                            f'line: {line.shape}, line_mean: {line_mean.shape}, line_var: {line_var.shape}'
                            )
                        print(f'Max line_var: {line_var.max()}')
                        ax.plot(line_mean, c=color, label=label, alpha=0.8)
                    else:
                        ax.plot(line_mean, c=color, alpha=0.8)
                else:
                    line_mean: torch.Tensor = line.mean(0)
                    if i == 0:
                        print(
                            f'line: {line.shape}, line_mean: {line_mean.shape}'
                            )
                        ax.plot(line_mean, c=color, label=label, alpha=0.8)
                    else:
                        ax.plot(line_mean, c=color, alpha=0.8)
            elif len(line.shape) == 1:
                if is_plot_var:
                    raise ValueError(f'Should be 3 dimensions')
                elif i == 0:
                    ax.plot(line, c=color, label=label, alpha=0.8)
                else:
                    ax.plot(line, c=color, alpha=0.8)
            else:
                raise ValueError(f'Should be 3 or 2 dimensions')
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reconstruction Loss')
    plt.savefig(fig_name)
    plt.show()
