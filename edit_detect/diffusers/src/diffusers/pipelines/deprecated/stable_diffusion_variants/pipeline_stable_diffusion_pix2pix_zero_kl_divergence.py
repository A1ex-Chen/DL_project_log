def kl_divergence(self, hidden_states):
    mean = hidden_states.mean()
    var = hidden_states.var()
    return var + mean ** 2 - 1 - torch.log(var + 1e-07)
