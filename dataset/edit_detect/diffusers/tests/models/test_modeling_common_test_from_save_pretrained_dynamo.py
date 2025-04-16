@require_python39_or_higher
@require_torch_2
@unittest.skipIf(get_python_version == (3, 12), reason=
    "Torch Dynamo isn't yet supported for Python 3.12.")
def test_from_save_pretrained_dynamo(self):
    init_dict, _ = self.prepare_init_args_and_inputs_for_common()
    inputs = [init_dict, self.model_class]
    run_test_in_subprocess(test_case=self, target_func=
        _test_from_save_pretrained_dynamo, inputs=inputs)
