def prepare_config_and_inputs_for_common(self):
    config_and_inputs = self.prepare_config_and_inputs()
    config, input_ids_1, input_ids_2, lm_labels = config_and_inputs
    inputs_dict = {'input_ids': input_ids_1}
    return config, inputs_dict
