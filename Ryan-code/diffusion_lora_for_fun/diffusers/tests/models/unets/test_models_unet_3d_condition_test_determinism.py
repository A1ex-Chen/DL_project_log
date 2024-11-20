def test_determinism(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        if torch_device == 'mps' and isinstance(model, ModelMixin):
            model(**self.dummy_input)
        first = model(**inputs_dict)
        if isinstance(first, dict):
            first = first.sample
        second = model(**inputs_dict)
        if isinstance(second, dict):
            second = second.sample
    out_1 = first.cpu().numpy()
    out_2 = second.cpu().numpy()
    out_1 = out_1[~np.isnan(out_1)]
    out_2 = out_2[~np.isnan(out_2)]
    max_diff = np.amax(np.abs(out_1 - out_2))
    self.assertLessEqual(max_diff, 1e-05)
