def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model
