def test_determinism(self, expected_max_diff=1e-05):
    if self.forward_requires_fresh_args:
        model = self.model_class(**self.init_dict)
    else:
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            first = model(**self.inputs_dict(0))
        else:
            first = model(**inputs_dict)
        if isinstance(first, dict):
            first = first.to_tuple()[0]
        if self.forward_requires_fresh_args:
            second = model(**self.inputs_dict(0))
        else:
            second = model(**inputs_dict)
        if isinstance(second, dict):
            second = second.to_tuple()[0]
    out_1 = first.cpu().numpy()
    out_2 = second.cpu().numpy()
    out_1 = out_1[~np.isnan(out_1)]
    out_2 = out_2[~np.isnan(out_2)]
    max_diff = np.amax(np.abs(out_1 - out_2))
    self.assertLessEqual(max_diff, expected_max_diff)
