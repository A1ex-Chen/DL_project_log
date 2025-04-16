@MODEL_WRAPPER_REGISTRY.register(model_name=model_name_key, dataset_name=
    'flowers102', task_type='classification', has_checkpoint=has_checkpoint)
def wrapper_func(pretrained=False, num_classes=FLOWERS102_NUM_CLASSES):
    wrapper_fn = MODEL_WRAPPER_REGISTRY.get(model_name=model_name_key,
        dataset_name='imagenet')
    model = wrapper_fn(pretrained=False, num_classes=num_classes)
    if pretrained:
        checkpoint_url = (
            f'{CHECKPOINT_STORAGE_URL}/{FLOWERS101_CHECKPOINT_URLS[model_name_key]}'
            )
        model = load_pretrained_weights(model, checkpoint_url)
    return model
