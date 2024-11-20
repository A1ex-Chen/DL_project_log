def kl_divergence(hidden_states):
    return hidden_states.var() + hidden_states.mean() ** 2 - 1 - torch.log(
        hidden_states.var() + 1e-07)
