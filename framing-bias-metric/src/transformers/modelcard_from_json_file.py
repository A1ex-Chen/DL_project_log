@classmethod
def from_json_file(cls, json_file):
    """Constructs a `ModelCard` from a json file of parameters."""
    with open(json_file, 'r', encoding='utf-8') as reader:
        text = reader.read()
    dict_obj = json.loads(text)
    return cls(**dict_obj)
