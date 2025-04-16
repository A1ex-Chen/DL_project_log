def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, 'w') as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)
