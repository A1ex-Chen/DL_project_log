def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path,
    mobilebert_config_file, pytorch_dump_path):
    config = MobileBertConfig.from_json_file(mobilebert_config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    model = MobileBertForPreTraining(config)
    model = load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
