def check_config_docstrings_have_checkpoints():
    configs_without_checkpoint = []
    for config_class in list(CONFIG_MAPPING.values()):
        checkpoint_found = False
        config_source = inspect.getsource(config_class)
        checkpoints = _re_checkpoint.findall(config_source)
        for checkpoint in checkpoints:
            ckpt_name, ckpt_link = checkpoint
            ckpt_link_from_name = f'https://huggingface.co/{ckpt_name}'
            if ckpt_link == ckpt_link_from_name:
                checkpoint_found = True
                break
        name = config_class.__name__
        if (not checkpoint_found and name not in
            CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK):
            configs_without_checkpoint.append(name)
    if len(configs_without_checkpoint) > 0:
        message = '\n'.join(sorted(configs_without_checkpoint))
        raise ValueError(
            f"""The following configurations don't contain any valid checkpoint:
{message}"""
            )
