def fit_config(server_round: int) ->Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    global DP
    config = {'local_epochs': 10, 'batch_size': 64, 'server_round':
        server_round, 'dp': DP}
    return config
