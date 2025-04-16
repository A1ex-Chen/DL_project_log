def initialize_hints_config(self, hints_file):
    if hints_file is None:
        file_to_open = deepview_profile.data.get_absolute_path('hints.yml')
    else:
        file_to_open = hints_file
    with open(file_to_open, 'r') as f:
        self.Hints = yaml.load(f, Loader=yaml.Loader)
