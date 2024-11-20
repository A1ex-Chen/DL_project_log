def main():
    num_client = CLIENT
    model = VAE(DATASET)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().
        items()]
    strategy = SaveModelStrategy(fraction_fit=1, fraction_evaluate=1,
        min_fit_clients=num_client, min_available_clients=num_client,
        min_evaluate_clients=num_client, evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config, on_evaluate_config_fn=evaluate_fig,
        initial_parameters=fl.common.ndarrays_to_parameters(
        model_parameters), evaluate_metrics_aggregation_fn=weighted_average)
    history = fl.server.History()
    num_rounds = 100
    history = fl.server.start_server(server_address='10.1.2.102:8080',
        config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy
        )
    np.save(
        f'Results/{DATASET}/{CLIENT}/{DP}/metrics/fid_loss_{num_rounds}.npy',
        history)
