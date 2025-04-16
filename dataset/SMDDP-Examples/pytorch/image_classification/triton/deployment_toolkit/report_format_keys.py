def format_keys(data: Dict) ->Dict:
    keys = {format_key(key=key): value for key, value in data.items()}
    return keys
