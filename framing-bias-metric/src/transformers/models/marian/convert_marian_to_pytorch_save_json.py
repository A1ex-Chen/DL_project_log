def save_json(content: Union[Dict, List], path: str) ->None:
    with open(path, 'w') as f:
        json.dump(content, f)
