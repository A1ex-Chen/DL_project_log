def convert_pt_checkpoint_to_tf(model_type, pytorch_checkpoint_path,
    config_file, tf_dump_path, compare_with_pt_model=False,
    use_cached_models=True):
    if model_type not in MODEL_CLASSES:
        raise ValueError('Unrecognized model type, should be one of {}.'.
            format(list(MODEL_CLASSES.keys())))
    (config_class, model_class, pt_model_class, aws_model_maps, aws_config_map
        ) = MODEL_CLASSES[model_type]
    if config_file in aws_config_map:
        config_file = cached_path(aws_config_map[config_file],
            force_download=not use_cached_models)
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print('Building TensorFlow model from configuration: {}'.format(str(
        config)))
    tf_model = model_class(config)
    if pytorch_checkpoint_path in aws_model_maps:
        pytorch_checkpoint_path = cached_path(aws_model_maps[
            pytorch_checkpoint_path], force_download=not use_cached_models)
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model,
        pytorch_checkpoint_path)
    if compare_with_pt_model:
        inputs_list = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        tf_inputs = tf.constant(inputs_list)
        tfo = tf_model(tf_inputs, training=False)
        pt_model = pt_model_class.from_pretrained(None, config=config,
            state_dict=torch.load(pytorch_checkpoint_path, map_location='cpu'))
        pt_inputs = torch.tensor(inputs_list)
        with torch.no_grad():
            pto = pt_model(pt_inputs)
        np_pt = pto[0].detach().numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print('Max absolute difference between models outputs {}'.format(diff))
        assert diff <= 0.02, 'Error, model absolute difference is >2e-2'
    print('Save TensorFlow model to {}'.format(tf_dump_path))
    tf_model.save_weights(tf_dump_path, save_format='h5')
