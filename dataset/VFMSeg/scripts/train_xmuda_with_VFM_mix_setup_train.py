def setup_train(model):
    if 'A' == model:
        model_A_2d.train()
        model_A_3d.train()
        train_metric_logger_A.reset()
    elif 'B' == model:
        model_B_2d.train()
        model_B_3d.train()
    else:
        raise ValueError('Unsupported type of Model Experiment Setting')
