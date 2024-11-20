def permute(w, skip_permute=skip_permute):
    if skip_permute:
        return w
    return w.view(n_heads, n_hidden // n_heads // 2, 2, n_hidden).transpose(
        1, 2).reshape(n_hidden, n_hidden)
