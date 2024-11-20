def save_json(data, name='data.json'):
    import json
    with open(name, 'w') as fp:
        json.dump(data, fp)
    return
