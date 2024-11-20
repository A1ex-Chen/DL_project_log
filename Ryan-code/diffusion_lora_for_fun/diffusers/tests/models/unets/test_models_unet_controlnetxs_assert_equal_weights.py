def assert_equal_weights(module, weight_dict_prefix):
    for param_name, param_value in module.named_parameters():
        assert torch.equal(model_state_dict[weight_dict_prefix + '.' +
            param_name], param_value)
