@unittest.skipIf(torch_device != 'cuda' or not is_accelerate_available() or
    is_accelerate_version('<', '0.14.0'), reason=
    'CPU offload is only available with CUDA and `accelerate v0.14.0` or higher'
    )
def test_sequential_offload_forward_pass_twice(self, expected_max_diff=0.0002):
    import accelerate
    generator_device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_sequential_cpu_offload()
    inputs = self.get_dummy_inputs(generator_device)
    output_with_offload = pipe(**inputs)[0]
    pipe.enable_sequential_cpu_offload()
    inputs = self.get_dummy_inputs(generator_device)
    output_with_offload_twice = pipe(**inputs)[0]
    max_diff = np.abs(to_np(output_with_offload) - to_np(
        output_with_offload_twice)).max()
    self.assertLess(max_diff, expected_max_diff,
        'running sequential offloading second time should have the inference results'
        )
    offloaded_modules = {k: v for k, v in pipe.components.items() if 
        isinstance(v, torch.nn.Module) and k not in pipe.
        _exclude_from_cpu_offload}
    self.assertTrue(all(v.device.type == 'meta' for v in offloaded_modules.
        values()),
        f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'meta']}"
        )
    self.assertTrue(all(hasattr(v, '_hf_hook') for k, v in
        offloaded_modules.items()),
        f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}"
        )
    offloaded_modules_with_incorrect_hooks = {}
    for k, v in offloaded_modules.items():
        if hasattr(v, '_hf_hook'):
            if isinstance(v._hf_hook, accelerate.hooks.SequentialHook):
                for hook in v._hf_hook.hooks:
                    if not isinstance(hook, accelerate.hooks.AlignDevicesHook):
                        offloaded_modules_with_incorrect_hooks[k] = type(v.
                            _hf_hook.hooks[0])
            elif not isinstance(v._hf_hook, accelerate.hooks.AlignDevicesHook):
                offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)
    self.assertTrue(len(offloaded_modules_with_incorrect_hooks) == 0,
        f'Not installed correct hook: {offloaded_modules_with_incorrect_hooks}'
        )
