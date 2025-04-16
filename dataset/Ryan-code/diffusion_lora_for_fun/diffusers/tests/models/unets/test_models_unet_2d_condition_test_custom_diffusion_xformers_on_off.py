@unittest.skipIf(torch_device != 'cuda' or not is_xformers_available(),
    reason=
    'XFormers attention is only available with CUDA and `xformers` installed')
def test_custom_diffusion_xformers_on_off(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 16, 32
    init_dict['attention_head_dim'] = 8, 16
    torch.manual_seed(0)
    model = self.model_class(**init_dict)
    model.to(torch_device)
    custom_diffusion_attn_procs = create_custom_diffusion_layers(model,
        mock_weights=False)
    model.set_attn_processor(custom_diffusion_attn_procs)
    with torch.no_grad():
        sample = model(**inputs_dict).sample
        model.enable_xformers_memory_efficient_attention()
        on_sample = model(**inputs_dict).sample
        model.disable_xformers_memory_efficient_attention()
        off_sample = model(**inputs_dict).sample
    assert (sample - on_sample).abs().max() < 0.0001
    assert (sample - off_sample).abs().max() < 0.0001
