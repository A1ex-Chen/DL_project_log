def test_model_attention_slicing(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 16, 32
    init_dict['attention_head_dim'] = 8
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    model.set_attention_slice('auto')
    with torch.no_grad():
        output = model(**inputs_dict)
    assert output is not None
    model.set_attention_slice('max')
    with torch.no_grad():
        output = model(**inputs_dict)
    assert output is not None
    model.set_attention_slice(2)
    with torch.no_grad():
        output = model(**inputs_dict)
    assert output is not None
