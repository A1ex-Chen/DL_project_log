def x_y_set_list_gen(df: pd.DataFrame, x_label: str, y_label: str, real_n:
    int, out_dist_n: int, fake_n: int):
    x_set_list = [torch.tensor(df[x_label][:real_n].values), torch.tensor(
        df[x_label][real_n:real_n + out_dist_n].values), torch.tensor(df[
        x_label][real_n + out_dist_n:].values)]
    y_set_list = [torch.tensor(df[y_label][:real_n].values), torch.tensor(
        df[y_label][real_n:real_n + out_dist_n].values), torch.tensor(df[
        y_label][real_n + out_dist_n:].values)]
    return x_set_list, y_set_list
