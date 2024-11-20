def save_to_json(self, json_path: str):
    """ Save the content of this instance in JSON format inside :obj:`json_path`."""
    json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True
        ) + '\n'
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_string)
