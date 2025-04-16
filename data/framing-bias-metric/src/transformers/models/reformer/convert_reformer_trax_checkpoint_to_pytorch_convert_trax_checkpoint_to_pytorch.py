def convert_trax_checkpoint_to_pytorch(trax_model_pkl_path, config_file,
    pytorch_dump_path):
    config = ReformerConfig.from_json_file(config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    model = ReformerModelWithLMHead(config)
    with open(trax_model_pkl_path, 'rb') as f:
        model_weights = pickle.load(f)['weights']
    set_model_weights_in_torch(model_weights, model, config.hidden_size)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
