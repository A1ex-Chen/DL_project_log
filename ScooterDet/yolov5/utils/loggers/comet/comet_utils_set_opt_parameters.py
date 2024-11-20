def set_opt_parameters(opt, experiment):
    """Update the opts Namespace with parameters
    from Comet's ExistingExperiment when resuming a run

    Args:
        opt (argparse.Namespace): Namespace of command line options
        experiment (comet_ml.APIExperiment): Comet API Experiment object
    """
    asset_list = experiment.get_asset_list()
    resume_string = opt.resume
    for asset in asset_list:
        if asset['fileName'] == 'opt.yaml':
            asset_id = asset['assetId']
            asset_binary = experiment.get_asset(asset_id, return_type=
                'binary', stream=False)
            opt_dict = yaml.safe_load(asset_binary)
            for key, value in opt_dict.items():
                setattr(opt, key, value)
            opt.resume = resume_string
    save_dir = f'{opt.project}/{experiment.name}'
    os.makedirs(save_dir, exist_ok=True)
    hyp_yaml_path = f'{save_dir}/hyp.yaml'
    with open(hyp_yaml_path, 'w') as f:
        yaml.dump(opt.hyp, f)
    opt.hyp = hyp_yaml_path
