def prepare_config_and_inputs_for_common(self):
    config_and_inputs = self.prepare_config_and_inputs()
    (config, input_ids, token_type_ids, input_lengths, sequence_labels,
        token_labels, is_impossible_labels, input_mask) = config_and_inputs
    inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids,
        'lengths': input_lengths}
    return config, inputs_dict
