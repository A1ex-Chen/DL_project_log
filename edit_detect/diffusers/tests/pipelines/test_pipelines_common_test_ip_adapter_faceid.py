def test_ip_adapter_faceid(self, expected_max_diff: float=0.0001):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components).to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    cross_attention_dim = pipe.unet.config.get('cross_attention_dim', 32)
    inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(
        torch_device))
    output_without_adapter = pipe(**inputs)[0]
    output_without_adapter = output_without_adapter[0, -3:, -3:, -1].flatten()
    adapter_state_dict = create_ip_adapter_faceid_state_dict(pipe.unet)
    pipe.unet._load_ip_adapter_weights(adapter_state_dict)
    inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(
        torch_device))
    inputs['ip_adapter_image_embeds'] = [self.
        _get_dummy_faceid_image_embeds(cross_attention_dim)]
    pipe.set_ip_adapter_scale(0.0)
    output_without_adapter_scale = pipe(**inputs)[0]
    output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1
        ].flatten()
    inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(
        torch_device))
    inputs['ip_adapter_image_embeds'] = [self.
        _get_dummy_faceid_image_embeds(cross_attention_dim)]
    pipe.set_ip_adapter_scale(42.0)
    output_with_adapter_scale = pipe(**inputs)[0]
    output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1
        ].flatten()
    max_diff_without_adapter_scale = np.abs(output_without_adapter_scale -
        output_without_adapter).max()
    max_diff_with_adapter_scale = np.abs(output_with_adapter_scale -
        output_without_adapter).max()
    self.assertLess(max_diff_without_adapter_scale, expected_max_diff,
        'Output without ip-adapter must be same as normal inference')
    self.assertGreater(max_diff_with_adapter_scale, 0.001,
        'Output with ip-adapter must be different from normal inference')
