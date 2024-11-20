def fetch_plot_run(df: pd.DataFrame, md: ModelDataset, x_col: str, y_col:
    str, output_fig_folder: str, file_name: str, real_n: int, out_dist_n:
    int, fake_n: int, ts: int, n: int):
    x_set_list, y_set_list = x_y_set_list_gen(df=df, x_label=x_col, y_label
        =y_col, real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_run(x_set_list=x_set_list, fig_name=os.path.join(output_fig_folder,
        f'{md.name_prefix}{file_name}_ts{ts}_n{n}.jpg'), title=
        f'Direct Reconstruction at Timestep: {ts} & {n} Samples',
        is_plot_var=False)
