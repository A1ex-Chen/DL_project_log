def write_json(my_dict, fname):
    json_str = json.dumps(my_dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)
