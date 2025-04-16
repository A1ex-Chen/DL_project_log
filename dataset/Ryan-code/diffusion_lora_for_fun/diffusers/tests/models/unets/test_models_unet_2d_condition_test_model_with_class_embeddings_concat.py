def test_model_with_class_embeddings_concat(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    batch_size, _, _, sample_size = inputs_dict['sample'].shape
    init_dict['class_embed_type'] = 'simple_projection'
    init_dict['projection_class_embeddings_input_dim'] = sample_size
    init_dict['class_embeddings_concat'] = True
    inputs_dict['class_labels'] = floats_tensor((batch_size, sample_size)).to(
        torch_device)
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        output = model(**inputs_dict)
        if isinstance(output, dict):
            output = output.sample
    self.assertIsNotNone(output)
    expected_shape = inputs_dict['sample'].shape
    self.assertEqual(output.shape, expected_shape,
        'Input and output shapes do not match')
