def save(self):
    """Save log_messages to file"""
    path = self.global_params['output_dir'
        ] if 'output_dir' in self.global_params else '.'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = '/run.{}.json'.format(self.run_id)
    with open(path + filename, 'a') as file_run_json:
        file_run_json.write(json.dumps(self.log_messages, indent=4,
            separators=(',', ': ')))
