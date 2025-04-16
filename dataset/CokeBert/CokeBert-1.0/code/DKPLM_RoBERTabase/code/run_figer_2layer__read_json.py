@classmethod
def _read_json(cls, input_file):
    with open(input_file, 'r') as f:
        return json.load(f)
