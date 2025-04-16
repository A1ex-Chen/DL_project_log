def list_models_by_dataset(dataset_name, with_checkpoint=False):
    return [model_key.model_name for model_key in list_models(dataset_name,
        print_table=False, with_checkpoint=with_checkpoint) if model_key.
        dataset_name == dataset_name]
