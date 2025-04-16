def test_from_save_pretrained_variant(self, expected_max_diff=5e-05):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    torch.manual_seed(0)
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_pretrained(tmpdirname, variant='fp16',
            safe_serialization=False)
        torch.manual_seed(0)
        new_model = self.model_class.from_pretrained(tmpdirname, variant='fp16'
            )
        with self.assertRaises(OSError) as error_context:
            self.model_class.from_pretrained(tmpdirname)
        assert 'Error no file named diffusion_pytorch_model.bin found in directory' in str(
            error_context.exception)
        new_model.to(torch_device)
    with torch.no_grad():
        image = model(**inputs_dict)
        if isinstance(image, dict):
            image = image.to_tuple()[0]
        new_image = new_model(**inputs_dict)
        if isinstance(new_image, dict):
            new_image = new_image.to_tuple()[0]
    max_diff = (image - new_image).abs().max().item()
    self.assertLessEqual(max_diff, expected_max_diff,
        'Models give different forward passes')
