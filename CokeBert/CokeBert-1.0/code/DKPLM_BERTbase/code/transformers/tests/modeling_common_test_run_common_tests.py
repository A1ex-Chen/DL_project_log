def run_common_tests(self, test_presents=False):
    config_and_inputs = self.prepare_config_and_inputs()
    self.create_and_check_base_model(*config_and_inputs)
    config_and_inputs = self.prepare_config_and_inputs()
    self.create_and_check_lm_head(*config_and_inputs)
    config_and_inputs = self.prepare_config_and_inputs()
    self.create_and_check_double_heads(*config_and_inputs)
    if test_presents:
        config_and_inputs = self.prepare_config_and_inputs()
        self.create_and_check_presents(*config_and_inputs)
