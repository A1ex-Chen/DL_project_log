def create_and_test_config_to_json_string(self):
    config = self.config_class(**self.inputs_dict)
    obj = json.loads(config.to_json_string())
    for key, value in self.inputs_dict.items():
        self.parent.assertEqual(obj[key], value)
