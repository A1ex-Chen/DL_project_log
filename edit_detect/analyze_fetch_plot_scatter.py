def fetch_plot_scatter(df: pd.DataFrame, md: ModelDataset, x_col: str,
    y_col: str, x_axis: str, y_axis: str, output_fig_folder: str, file_name:
    str, real_n: int, out_dist_n: int, fake_n: int, ts: int, n: int, xscale:
    str='linear', yscale: str='linear'):
    x_set_list, y_set_list = x_y_set_list_gen(df=df, x_label=x_col, y_label
        =y_col, real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=
        f'{x_axis} & {y_axis} at Timestep: {ts} & {n} Samples', fig_name=os
        .path.join(output_fig_folder,
        f'{md.name_prefix}{file_name}_ts{ts}_n{n}'), xlabel=x_axis, ylabel=
        y_axis, xscale=xscale, yscale=yscale)
