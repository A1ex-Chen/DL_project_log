def _expand_time(m):
    mins = int(m.group(2))
    if mins == 0:
        return _inflect.number_to_words(m.group(1))
    return ' '.join([_inflect.number_to_words(m.group(1)), _inflect.
        number_to_words(m.group(2))])
