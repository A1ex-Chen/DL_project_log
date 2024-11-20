@require_python39_or_higher
@require_torch_2
def test_stable_diffusion_compile(self):
    run_test_in_subprocess(test_case=self, target_func=
        _test_stable_diffusion_compile, inputs=None)
