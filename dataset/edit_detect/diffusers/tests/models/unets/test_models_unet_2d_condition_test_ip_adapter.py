def test_ip_adapter(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 16, 32
    init_dict['attention_head_dim'] = 8, 16
    model = self.model_class(**init_dict)
    model.to(torch_device)
    with torch.no_grad():
        sample1 = model(**inputs_dict).sample
    batch_size = inputs_dict['encoder_hidden_states'].shape[0]
    image_embeds = floats_tensor((batch_size, 1, model.config.
        cross_attention_dim)).to(torch_device)
    inputs_dict['added_cond_kwargs'] = {'image_embeds': [image_embeds]}
    ip_adapter_1 = create_ip_adapter_state_dict(model)
    image_proj_state_dict_2 = {k: (w + 1.0) for k, w in ip_adapter_1[
        'image_proj'].items()}
    cross_attn_state_dict_2 = {k: (w + 1.0) for k, w in ip_adapter_1[
        'ip_adapter'].items()}
    ip_adapter_2 = {}
    ip_adapter_2.update({'image_proj': image_proj_state_dict_2,
        'ip_adapter': cross_attn_state_dict_2})
    model._load_ip_adapter_weights([ip_adapter_1])
    assert model.config.encoder_hid_dim_type == 'ip_image_proj'
    assert model.encoder_hid_proj is not None
    assert model.down_blocks[0].attentions[0].transformer_blocks[0
        ].attn2.processor.__class__.__name__ in ('IPAdapterAttnProcessor',
        'IPAdapterAttnProcessor2_0')
    with torch.no_grad():
        sample2 = model(**inputs_dict).sample
    model._load_ip_adapter_weights([ip_adapter_2])
    with torch.no_grad():
        sample3 = model(**inputs_dict).sample
    model._load_ip_adapter_weights([ip_adapter_1])
    with torch.no_grad():
        sample4 = model(**inputs_dict).sample
    model._load_ip_adapter_weights([ip_adapter_1, ip_adapter_2])
    for attn_processor in model.attn_processors.values():
        if isinstance(attn_processor, (IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_0)):
            attn_processor.scale = [1, 0]
    image_embeds_multi = image_embeds.repeat(1, 2, 1)
    inputs_dict['added_cond_kwargs'] = {'image_embeds': [image_embeds_multi,
        image_embeds_multi]}
    with torch.no_grad():
        sample5 = model(**inputs_dict).sample
    image_embeds = image_embeds.squeeze(1)
    inputs_dict['added_cond_kwargs'] = {'image_embeds': image_embeds}
    model._load_ip_adapter_weights(ip_adapter_1)
    with torch.no_grad():
        sample6 = model(**inputs_dict).sample
    assert not sample1.allclose(sample2, atol=0.0001, rtol=0.0001)
    assert not sample2.allclose(sample3, atol=0.0001, rtol=0.0001)
    assert sample2.allclose(sample4, atol=0.0001, rtol=0.0001)
    assert sample2.allclose(sample5, atol=0.0001, rtol=0.0001)
    assert sample2.allclose(sample6, atol=0.0001, rtol=0.0001)
