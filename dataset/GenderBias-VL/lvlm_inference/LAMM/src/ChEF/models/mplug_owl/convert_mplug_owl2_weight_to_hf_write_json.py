def write_json(text, path):
    with open(path, 'w') as f:
        json.dump(text, f)
