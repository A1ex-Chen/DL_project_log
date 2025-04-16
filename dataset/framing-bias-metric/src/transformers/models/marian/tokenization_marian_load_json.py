def load_json(path: str) ->Union[Dict, List]:
    with open(path, 'r') as f:
        return json.load(f)
