def prepare_config_and_inputs_for_common(self):
    config_and_inputs = self.prepare_config_and_inputs()
    (config, input_ids, input_mask, head_mask, token_type_ids, mc_token_ids,
        sequence_labels, token_labels, choice_labels) = config_and_inputs
    inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids,
        'head_mask': head_mask}
    return config, inputs_dict
