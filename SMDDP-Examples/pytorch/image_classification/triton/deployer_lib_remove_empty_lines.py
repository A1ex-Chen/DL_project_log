def remove_empty_lines(text):
    """ removes empty lines from text, returns the result """
    ret = ''.join([s for s in text.strip().splitlines(True) if s.strip()])
    return ret
