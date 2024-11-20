def test_model_from_pretrained(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_pretrained(tmpdirname, safe_serialization=False)
        new_model = self.model_class.from_pretrained(tmpdirname)
        new_model.to(torch_device)
        new_model.eval()
    for param_name in model.state_dict().keys():
        param_1 = model.state_dict()[param_name]
        param_2 = new_model.state_dict()[param_name]
        self.assertEqual(param_1.shape, param_2.shape)
    with torch.no_grad():
        output_1 = model(**inputs_dict)
        if isinstance(output_1, dict):
            output_1 = output_1.to_tuple()[0]
        output_2 = new_model(**inputs_dict)
        if isinstance(output_2, dict):
            output_2 = output_2.to_tuple()[0]
    self.assertEqual(output_1.shape, output_2.shape)
