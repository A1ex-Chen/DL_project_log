def truncate(body_text):
    tokens = body_text.split(' ')
    return ' '.join(tokens[:250])
