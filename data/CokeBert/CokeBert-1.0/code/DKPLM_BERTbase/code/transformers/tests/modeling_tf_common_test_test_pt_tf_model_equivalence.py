def test_pt_tf_model_equivalence(self):
    if not is_torch_available():
        return
    import torch
    import transformers
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        pt_model_class_name = model_class.__name__[2:]
        pt_model_class = getattr(transformers, pt_model_class_name)
        config.output_hidden_states = True
        tf_model = model_class(config)
        pt_model = pt_model_class(config)
        tf_model = transformers.load_pytorch_model_in_tf2_model(tf_model,
            pt_model, tf_inputs=inputs_dict)
        pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model,
            tf_model)
        pt_model.eval()
        pt_inputs_dict = dict((name, torch.from_numpy(key.numpy()).to(torch
            .long)) for name, key in inputs_dict.items())
        with torch.no_grad():
            pto = pt_model(**pt_inputs_dict)
        tfo = tf_model(inputs_dict)
        max_diff = np.amax(np.abs(tfo[0].numpy() - pto[0].numpy()))
        self.assertLessEqual(max_diff, 0.02)
        with TemporaryDirectory() as tmpdirname:
            pt_checkpoint_path = os.path.join(tmpdirname, 'pt_model.bin')
            torch.save(pt_model.state_dict(), pt_checkpoint_path)
            tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(
                tf_model, pt_checkpoint_path)
            tf_checkpoint_path = os.path.join(tmpdirname, 'tf_model.h5')
            tf_model.save_weights(tf_checkpoint_path)
            pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(
                pt_model, tf_checkpoint_path)
        pt_model.eval()
        pt_inputs_dict = dict((name, torch.from_numpy(key.numpy()).to(torch
            .long)) for name, key in inputs_dict.items())
        with torch.no_grad():
            pto = pt_model(**pt_inputs_dict)
        tfo = tf_model(inputs_dict)
        max_diff = np.amax(np.abs(tfo[0].numpy() - pto[0].numpy()))
        self.assertLessEqual(max_diff, 0.02)
