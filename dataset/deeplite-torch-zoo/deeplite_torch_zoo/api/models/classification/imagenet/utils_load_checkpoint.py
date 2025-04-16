def load_checkpoint(model, model_name, dataset_name, model_urls, device='cpu'):
    if f'{model_name}_{dataset_name}' not in model_urls:
        raise ValueError(
            f"""Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. 
Use pretrained=False if you want to create a untrained model."""
            )
    return load_pretrained_weights(model, model_urls[
        f'{model_name}_{dataset_name}'], device)
