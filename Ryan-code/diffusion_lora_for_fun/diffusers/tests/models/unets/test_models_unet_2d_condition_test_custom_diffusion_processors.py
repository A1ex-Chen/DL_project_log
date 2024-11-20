def test_custom_diffusion_processors(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 16, 32
    init_dict['attention_head_dim'] = 8, 16
    model = self.model_class(**init_dict)
    model.to(torch_device)
    with torch.no_grad():
        sample1 = model(**inputs_dict).sample
    custom_diffusion_attn_procs = create_custom_diffusion_layers(model,
        mock_weights=False)
    model.set_attn_processor(custom_diffusion_attn_procs)
    model.to(torch_device)
    model.set_attn_processor(model.attn_processors)
    with torch.no_grad():
        sample2 = model(**inputs_dict).sample
    assert (sample1 - sample2).abs().max() < 0.003
