def fit_config(server_round: int) ->Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {'local_epochs': l_local, 'batch_size': batch_size,
        'server_round': server_round, 'dp': DP}
    return config
