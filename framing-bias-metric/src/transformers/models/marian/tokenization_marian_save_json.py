def save_json(data, path: str) ->None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
