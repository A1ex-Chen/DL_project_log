def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    drop_idx = []
    for i, char in enumerate(text):
        cp = ord(char)
        if cp == 0 or cp == 65533 or _is_control(char):
            drop_idx.append(i)
            continue
        if _is_whitespace(char):
            output.append(' ')
        else:
            output.append(char)
    return ''.join(output), drop_idx
