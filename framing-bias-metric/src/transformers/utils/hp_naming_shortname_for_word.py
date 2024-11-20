@staticmethod
def shortname_for_word(info, word):
    if len(word) == 0:
        return ''
    short_word = None
    if any(char.isdigit() for char in word):
        raise Exception(
            f"Parameters should not contain numbers: '{word}' contains a number"
            )
    if word in info['short_word']:
        return info['short_word'][word]
    for prefix_len in range(1, len(word) + 1):
        prefix = word[:prefix_len]
        if prefix in info['reverse_short_word']:
            continue
        else:
            short_word = prefix
            break
    if short_word is None:

        def int_to_alphabetic(integer):
            s = ''
            while integer != 0:
                s = chr(ord('A') + integer % 10) + s
                integer //= 10
            return s
        i = 0
        while True:
            sword = word + '#' + int_to_alphabetic(i)
            if sword in info['reverse_short_word']:
                continue
            else:
                short_word = sword
                break
    info['short_word'][word] = short_word
    info['reverse_short_word'][short_word] = word
    return short_word
