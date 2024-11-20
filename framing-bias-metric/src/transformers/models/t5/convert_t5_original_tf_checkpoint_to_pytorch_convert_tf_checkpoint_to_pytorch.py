def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file,
    pytorch_dump_path):
    config = T5Config.from_json_file(config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    model = T5ForConditionalGeneration(config)
    load_tf_weights_in_t5(model, config, tf_checkpoint_path)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    model.save_pretrained(pytorch_dump_path)
