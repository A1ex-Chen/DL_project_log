def split_ctrl_code(text):
    r = re.search('\\[(?P<code>[a-z]+)\\](?P<text>.+)', text)
    if r:
        return r.group('code').strip(), r.group('text').strip()
    return '', text
