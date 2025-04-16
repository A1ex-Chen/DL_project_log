def load_pretrained_model(model, model_name, dataset_name, device='cpu'):
    if f'{model_name}_{dataset_name}' not in model_urls:
        raise ValueError(
            f"""Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. 
Use pretrained=False if you want to create a untrained model."""
            )
    checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[
        f'{model_name}_{dataset_name}'])
    model = load_pretrained_weights(model, checkpoint_url, device)
    return model
