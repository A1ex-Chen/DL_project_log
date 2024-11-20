@unittest.skip(reason=
    'Skip torch.compile test to speed up the slow test suite.')
@require_torch_2
def test_unidiffuser_compile(self, seed=0):
    inputs = self.get_inputs(torch_device, seed=seed, generate_latents=True)
    del inputs['prompt']
    del inputs['image']
    del inputs['generator']
    inputs['torch_device'] = torch_device
    inputs['seed'] = seed
    run_test_in_subprocess(test_case=self, target_func=
        _test_unidiffuser_compile, inputs=inputs)
