def load_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
        return data
