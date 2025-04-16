def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file,
    pytorch_dump_path, discriminator_or_generator):
    config = ElectraConfig.from_json_file(config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    if discriminator_or_generator == 'discriminator':
        model = ElectraForPreTraining(config)
    elif discriminator_or_generator == 'generator':
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError(
            "The discriminator_or_generator argument should be either 'discriminator' or 'generator'"
            )
    load_tf_weights_in_electra(model, config, tf_checkpoint_path,
        discriminator_or_generator=discriminator_or_generator)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
