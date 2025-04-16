def test_custom_diffusion_save_load(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 16, 32
    init_dict['attention_head_dim'] = 8, 16
    torch.manual_seed(0)
    model = self.model_class(**init_dict)
    model.to(torch_device)
    with torch.no_grad():
        old_sample = model(**inputs_dict).sample
    custom_diffusion_attn_procs = create_custom_diffusion_layers(model,
        mock_weights=False)
    model.set_attn_processor(custom_diffusion_attn_procs)
    with torch.no_grad():
        sample = model(**inputs_dict).sample
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_attn_procs(tmpdirname, safe_serialization=False)
        self.assertTrue(os.path.isfile(os.path.join(tmpdirname,
            'pytorch_custom_diffusion_weights.bin')))
        torch.manual_seed(0)
        new_model = self.model_class(**init_dict)
        new_model.load_attn_procs(tmpdirname, weight_name=
            'pytorch_custom_diffusion_weights.bin')
        new_model.to(torch_device)
    with torch.no_grad():
        new_sample = new_model(**inputs_dict).sample
    assert (sample - new_sample).abs().max() < 0.0001
    assert (sample - old_sample).abs().max() < 0.003
