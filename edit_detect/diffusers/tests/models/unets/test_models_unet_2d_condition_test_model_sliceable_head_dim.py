def test_model_sliceable_head_dim(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 16, 32
    init_dict['attention_head_dim'] = 8, 16
    model = self.model_class(**init_dict)

    def check_sliceable_dim_attr(module: torch.nn.Module):
        if hasattr(module, 'set_attention_slice'):
            assert isinstance(module.sliceable_head_dim, int)
        for child in module.children():
            check_sliceable_dim_attr(child)
    for module in model.children():
        check_sliceable_dim_attr(module)
