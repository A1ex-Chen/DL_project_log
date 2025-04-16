def load_json(name):
    import json
    with open(name, 'r') as fp:
        data = json.load(fp)
    return data
