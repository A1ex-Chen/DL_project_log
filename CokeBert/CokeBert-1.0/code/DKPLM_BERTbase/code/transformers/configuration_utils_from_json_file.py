@classmethod
def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with open(json_file, 'r', encoding='utf-8') as reader:
        text = reader.read()
    return cls.from_dict(json.loads(text))
