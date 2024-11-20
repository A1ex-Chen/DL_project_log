def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)
