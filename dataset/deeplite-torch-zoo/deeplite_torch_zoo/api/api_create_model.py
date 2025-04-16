@deprecated
def create_model(model_name, pretraining_dataset, num_classes=None,
    pretrained=False, **kwargs):
    return get_model(model_name=model_name, dataset_name=
        pretraining_dataset, num_classes=num_classes, pretrained=pretrained,
        **kwargs)
