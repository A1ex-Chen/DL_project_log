@staticmethod
def fix_yaml(path):
    """
        Function to fix YAML train and val path.

        Args:
            path (str): YAML file path.
        """
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data['train'] = 'train/images'
    yaml_data['val'] = 'valid/images'
    with open(path, 'w') as file:
        yaml.safe_dump(yaml_data, file)
