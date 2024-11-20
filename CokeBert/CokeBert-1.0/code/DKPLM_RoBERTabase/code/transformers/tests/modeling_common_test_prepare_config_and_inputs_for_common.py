def prepare_config_and_inputs_for_common(self):
    config_and_inputs = self.prepare_config_and_inputs()
    (config, input_ids, token_type_ids, position_ids, mc_labels, lm_labels,
        mc_token_ids) = config_and_inputs
    inputs_dict = {'input_ids': input_ids}
    return config, inputs_dict
