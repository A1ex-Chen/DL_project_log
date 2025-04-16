def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize('NFD', text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == 'Mn':
            continue
        output.append(char)
    return ''.join(output)
