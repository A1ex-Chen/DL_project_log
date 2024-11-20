@require_python39_or_higher
@require_torch_2
@unittest.skipIf(get_python_version == (3, 12), reason=
    "Torch Dynamo isn't yet supported for Python 3.12.")
def test_stable_diffusion_compile(self):
    run_test_in_subprocess(test_case=self, target_func=
        _test_stable_diffusion_compile, inputs=None)
