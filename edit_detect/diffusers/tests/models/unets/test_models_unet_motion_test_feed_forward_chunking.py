def test_feed_forward_chunking(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 32, 64
    init_dict['norm_num_groups'] = 32
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        output = model(**inputs_dict)[0]
    model.enable_forward_chunking()
    with torch.no_grad():
        output_2 = model(**inputs_dict)[0]
    self.assertEqual(output.shape, output_2.shape, "Shape doesn't match")
    assert np.abs(output.cpu() - output_2.cpu()).max() < 0.01
