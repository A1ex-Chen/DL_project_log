def load_json(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    return data
