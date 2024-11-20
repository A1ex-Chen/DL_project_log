def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    if (cp >= 33 and cp <= 47 or cp >= 58 and cp <= 64 or cp >= 91 and cp <=
        96 or cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False
