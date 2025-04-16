def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file,
    pytorch_dump_path):
    config = FunnelConfig.from_json_file(config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    model = FunnelForPreTraining(config)
    load_tf_weights_in_funnel(model, config, tf_checkpoint_path)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
