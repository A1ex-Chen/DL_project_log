def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    if (cp >= 19968 and cp <= 40959 or cp >= 13312 and cp <= 19903 or cp >=
        131072 and cp <= 173791 or cp >= 173824 and cp <= 177983 or cp >= 
        177984 and cp <= 178207 or cp >= 178208 and cp <= 183983 or cp >= 
        63744 and cp <= 64255 or cp >= 194560 and cp <= 195103):
        return True
    return False
