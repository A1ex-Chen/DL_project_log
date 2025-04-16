def test_output(self, expected_slice):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    unet_block = self.block_class(**init_dict)
    unet_block.to(torch_device)
    unet_block.eval()
    with torch.no_grad():
        output = unet_block(**inputs_dict)
    if isinstance(output, Tuple):
        output = output[0]
    self.assertEqual(output.shape, self.output_shape)
    output_slice = output[0, -1, -3:, -3:]
    expected_slice = torch.tensor(expected_slice).to(torch_device)
    assert torch_all_close(output_slice.flatten(), expected_slice, atol=0.005)
