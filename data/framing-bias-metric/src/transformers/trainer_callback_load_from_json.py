@classmethod
def load_from_json(cls, json_path: str):
    """ Create an instance from the content of :obj:`json_path`."""
    with open(json_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return cls(**json.loads(text))
