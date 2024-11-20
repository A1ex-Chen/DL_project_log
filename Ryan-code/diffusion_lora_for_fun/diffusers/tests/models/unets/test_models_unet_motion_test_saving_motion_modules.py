def test_saving_motion_modules(self):
    torch.manual_seed(0)
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.to(torch_device)
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_motion_modules(tmpdirname)
        self.assertTrue(os.path.isfile(os.path.join(tmpdirname,
            'diffusion_pytorch_model.safetensors')))
        adapter_loaded = MotionAdapter.from_pretrained(tmpdirname)
        torch.manual_seed(0)
        model_loaded = self.model_class(**init_dict)
        model_loaded.load_motion_modules(adapter_loaded)
        model_loaded.to(torch_device)
    with torch.no_grad():
        output = model(**inputs_dict)[0]
        output_loaded = model_loaded(**inputs_dict)[0]
    max_diff = (output - output_loaded).abs().max().item()
    self.assertLessEqual(max_diff, 0.0001,
        'Models give different forward passes')
