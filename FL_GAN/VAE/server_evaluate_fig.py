def evaluate_fig(server_round: int):
    val_batch_size = 64
    global DATASET, DP
    dataset = DATASET
    dp = DP
    val_config = {'val_batch_size': val_batch_size, 'server_round':
        server_round, 'dataset': dataset, 'dp': dp}
    return val_config
