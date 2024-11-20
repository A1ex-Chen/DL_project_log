def save_models(model_dir, models, global_step, max_to_keep=15, keep_latest
    =True):
    with DelayedKeyboardInterrupt():
        name_to_model = _get_name_to_model_map(models)
        for name, model in name_to_model.items():
            save(model_dir, model, name, global_step, max_to_keep, keep_latest)
