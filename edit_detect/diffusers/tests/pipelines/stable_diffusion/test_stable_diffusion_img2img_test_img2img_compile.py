@require_python39_or_higher
@require_torch_2
def test_img2img_compile(self):
    seed = 0
    inputs = self.get_inputs(torch_device, seed=seed)
    del inputs['generator']
    inputs['torch_device'] = torch_device
    inputs['seed'] = seed
    run_test_in_subprocess(test_case=self, target_func=
        _test_img2img_compile, inputs=inputs)
