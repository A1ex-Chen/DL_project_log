def json_load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data
