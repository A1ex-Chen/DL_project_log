def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 65533 or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(' ')
        else:
            output.append(char)
    return ''.join(output)
