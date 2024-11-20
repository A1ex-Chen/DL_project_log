def create_and_test_config_to_json_file(self):
    config_first = self.config_class(**self.inputs_dict)
    json_file_path = os.path.join(os.getcwd(), 'config_' + str(uuid.uuid4()
        ) + '.json')
    config_first.to_json_file(json_file_path)
    config_second = self.config_class.from_json_file(json_file_path)
    os.remove(json_file_path)
    self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())
