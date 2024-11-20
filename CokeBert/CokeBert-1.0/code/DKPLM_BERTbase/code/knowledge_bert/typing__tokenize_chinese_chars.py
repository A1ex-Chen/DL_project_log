def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if self._is_chinese_char(cp):
            output.append(' ')
            output.append(char)
            output.append(' ')
        else:
            output.append(char)
    return ''.join(output)
