def format_key(key: str) ->str:
    key = ' '.join([k.capitalize() for k in re.split('_| ', key)])
    return key
