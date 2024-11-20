def load(self):
    model_name = self.name
    model_path = self.path
    device = self.device
    verbose = self.verbose
    if model_name != '' and model_path == '':
        model_path = download_model(model_name)
        logger.debug(model_path)
    if model_name.endswith('MSMT17'):
        model_name = model_name.replace('_MSMT17', '')
    if verbose:
        return FeatureExtractor(model_name=model_name, model_path=
            model_path, device=device)
    with HiddenPrints():
        return FeatureExtractor(model_name=model_name, model_path=
            model_path, device=device)
