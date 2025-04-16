def get_token(cookie='./cookie'):
    with open(cookie) as f:
        for line in f:
            if 'download' in line:
                return line.split()[-1]
    return ''
