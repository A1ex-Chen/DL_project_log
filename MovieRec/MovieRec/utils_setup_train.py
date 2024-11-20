def setup_train(args):
    set_up_gpu(args)
    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)
    pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return export_root
