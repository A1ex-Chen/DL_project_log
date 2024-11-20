@classmethod
def _read_json(cls, input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.loads(f.read())
