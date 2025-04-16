@unittest.skipIf(torch_device != 'cuda' or not is_xformers_available(),
    reason=
    'XFormers attention is only available with CUDA and `xformers` installed')
def test_xformers_attention_forwardGenerator_pass(self):
    expected_max_diff = 0.0009
    if not self.test_xformers_attention:
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    output_without_offload = pipe(**inputs).frames[0]
    output_without_offload = output_without_offload.cpu() if torch.is_tensor(
        output_without_offload) else output_without_offload
    pipe.enable_xformers_memory_efficient_attention()
    inputs = self.get_dummy_inputs(torch_device)
    output_with_offload = pipe(**inputs).frames[0]
    output_with_offload = output_with_offload.cpu() if torch.is_tensor(
        output_with_offload) else output_without_offload
    max_diff = np.abs(to_np(output_with_offload) - to_np(
        output_without_offload)).max()
    self.assertLess(max_diff, expected_max_diff,
        'XFormers attention should not affect the inference results')
