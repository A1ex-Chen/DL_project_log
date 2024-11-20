def test_ip_adapter_cfg(self, expected_max_diff: float=0.0001):
    parameters = inspect.signature(self.pipeline_class.__call__).parameters
    if 'guidance_scale' not in parameters:
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components).to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    cross_attention_dim = pipe.unet.config.get('cross_attention_dim', 32)
    adapter_state_dict = create_ip_adapter_state_dict(pipe.unet)
    pipe.unet._load_ip_adapter_weights(adapter_state_dict)
    pipe.set_ip_adapter_scale(1.0)
    inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(
        torch_device))
    inputs['ip_adapter_image_embeds'] = [self._get_dummy_image_embeds(
        cross_attention_dim)[0].unsqueeze(0)]
    inputs['guidance_scale'] = 1.0
    out_no_cfg = pipe(**inputs)[0]
    inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(
        torch_device))
    inputs['ip_adapter_image_embeds'] = [self._get_dummy_image_embeds(
        cross_attention_dim)]
    inputs['guidance_scale'] = 7.5
    out_cfg = pipe(**inputs)[0]
    assert out_cfg.shape == out_no_cfg.shape
