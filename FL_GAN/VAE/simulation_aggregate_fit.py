def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.
    client_proxy.ClientProxy, fl.common.FitRes]], failures: List[Union[
    Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes],
    BaseException]]) ->Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]
    ]:
    aggregated_parameters, aggregated_metrics = super().aggregate_fit(
        server_round, results, failures)
    global DATASET, CLIENT, DP
    if aggregated_parameters is not None:
        aggregated_ndarrays: List[np.ndarray
            ] = fl.common.parameters_to_ndarrays(aggregated_parameters)
        print(f'Saving round {server_round} aggregated_ndarrays...')
        np.savez(
            f'Results/{DATASET}/{CLIENT}/{DP}/weights/FL_round_{server_round}_weights.npz'
            , *aggregated_ndarrays)
    return aggregated_parameters, aggregated_metrics
