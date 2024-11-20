def create_model_and_transforms(model_name: str, pretrained: str='',
    precision: str='fp32', device: torch.device=torch.device('cpu'), jit:
    bool=False, force_quick_gelu: bool=False):
    model = create_model(model_name, pretrained, precision, device, jit,
        force_quick_gelu=force_quick_gelu)
    preprocess_train = image_transform(model.visual.image_size, is_train=True)
    preprocess_val = image_transform(model.visual.image_size, is_train=False)
    return model, preprocess_train, preprocess_val
