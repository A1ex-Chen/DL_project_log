def parse_bool(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise ValueError(f'could not parse string as bool {string}')
