def get_evaluate_fig(batch_size, DP, DATASET):

    def evaluate_fig(server_round: int):
        val_config = {'val_batch_size': batch_size, 'server_round':
            server_round, 'dataset': DATASET, 'dp': DP}
        return val_config
    return evaluate_fig
