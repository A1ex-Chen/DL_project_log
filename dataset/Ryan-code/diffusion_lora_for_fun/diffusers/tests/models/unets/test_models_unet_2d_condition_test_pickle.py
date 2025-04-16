def test_pickle(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 16, 32
    init_dict['attention_head_dim'] = 8, 16
    model = self.model_class(**init_dict)
    model.to(torch_device)
    with torch.no_grad():
        sample = model(**inputs_dict).sample
    sample_copy = copy.copy(sample)
    assert (sample - sample_copy).abs().max() < 0.0001
