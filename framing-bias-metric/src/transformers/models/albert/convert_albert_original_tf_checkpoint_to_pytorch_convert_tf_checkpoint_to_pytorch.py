def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, albert_config_file,
    pytorch_dump_path):
    config = AlbertConfig.from_json_file(albert_config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    model = AlbertForPreTraining(config)
    load_tf_weights_in_albert(model, config, tf_checkpoint_path)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
