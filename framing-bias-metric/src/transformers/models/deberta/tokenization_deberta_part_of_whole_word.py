def part_of_whole_word(self, token, is_bos=False):
    if is_bos:
        return True
    s = self._decode(token)
    if len(s) == 1 and (_is_whitespace(list(s)[0]) or _is_control(list(s)[0
        ]) or _is_punctuation(list(s)[0])):
        return False
    return not s.startswith(' ')
