def _is_whitespace(c):
    if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 8239:
        return True
    return False
