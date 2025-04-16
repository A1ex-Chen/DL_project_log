def import_model_class_from_model_name_or_path(pretrained_model_name_or_path:
    str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        revision)
    model_class = text_encoder_config.architectures[0]
    if model_class == 'CLIPTextModel':
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f'{model_class} is not supported.')
