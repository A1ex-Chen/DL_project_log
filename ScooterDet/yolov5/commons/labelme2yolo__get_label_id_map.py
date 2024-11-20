def _get_label_id_map(self, json_dir):
    label_set = set()
    for file_name in os.listdir(json_dir):
        if file_name.endswith('json'):
            json_path = os.path.join(json_dir, file_name)
            data = json.load(open(json_path))
            for shape in data['shapes']:
                label_set.add(shape['label'])
    return OrderedDict([(label, label_id) for label_id, label in enumerate(
        label_set)])
