@staticmethod
def reload_dataset(data, task='val'):
    with open(data, errors='ignore') as yaml_file:
        data = yaml.safe_load(yaml_file)
    task = 'test' if task == 'test' else 'val'
    path = data.get(task, 'val')
    if not isinstance(path, list):
        path = [path]
    for p in path:
        if not os.path.exists(p):
            raise Exception(f'Dataset path {p} not found.')
    return data
