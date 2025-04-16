@deprecated
def get_model_by_name(model_name, dataset_name, pretrained=True, **kwargs):
    return get_model(model_name=model_name, dataset_name=dataset_name,
        pretrained=pretrained, **kwargs)
