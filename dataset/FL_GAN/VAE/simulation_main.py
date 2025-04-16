def main(args):
    num_client = args.client
    DATASET = args.dataset
    DP = args.dp
    batch_size = args.batch_size
    l_epochs = args.l_epochs
    model = VAE(DATASET)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().
        items()]
    strategy = SaveModelStrategy(fraction_fit=1, fraction_evaluate=1,
        min_fit_clients=num_client, min_available_clients=num_client,
        min_evaluate_clients=num_client, evaluate_fn=get_evaluate_fn(args),
        on_fit_config_fn=get_fit_cofig(l_epochs, batch_size, DP),
        on_evaluate_config_fn=get_evaluate_fig(batch_size, DP, DATASET),
        initial_parameters=fl.common.ndarrays_to_parameters(
        model_parameters), evaluate_metrics_aggregation_fn=weighted_average)
    history = fl.server.History()
    num_rounds = args.g_epochs
    history = fl.simulation.start_simulation(client_fn=get_client_fn(args),
        num_clients=num_client, config=fl.server.ServerConfig(num_rounds=
        num_rounds), strategy=strategy)
    fid_save_path = f'Results/{DATASET}/{num_client}/{DP}/metrics'
    if not os.path.exists(fid_save_path):
        os.makedirss(fid_save_path)
    fid_save_path = fid_save_path + f'/fid_loss_{num_rounds}.npy'
    np.save(fid_save_path, history)
