def test_from_save_pretrained(self, expected_max_diff=5e-05):
    if self.forward_requires_fresh_args:
        model = self.model_class(**self.init_dict)
    else:
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
    if hasattr(model, 'set_default_attn_processor'):
        model.set_default_attn_processor()
    model.to(torch_device)
    model.eval()
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_pretrained(tmpdirname, safe_serialization=False)
        new_model = self.model_class.from_pretrained(tmpdirname)
        if hasattr(new_model, 'set_default_attn_processor'):
            new_model.set_default_attn_processor()
        new_model.to(torch_device)
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            image = model(**self.inputs_dict(0))
        else:
            image = model(**inputs_dict)
        if isinstance(image, dict):
            image = image.to_tuple()[0]
        if self.forward_requires_fresh_args:
            new_image = new_model(**self.inputs_dict(0))
        else:
            new_image = new_model(**inputs_dict)
        if isinstance(new_image, dict):
            new_image = new_image.to_tuple()[0]
    max_diff = (image - new_image).abs().max().item()
    self.assertLessEqual(max_diff, expected_max_diff,
        'Models give different forward passes')
