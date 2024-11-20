def reset_x0(self, x_in, cond, act_dim):
    for key, val in cond.items():
        x_in[:, key, act_dim:] = val.clone()
    return x_in
