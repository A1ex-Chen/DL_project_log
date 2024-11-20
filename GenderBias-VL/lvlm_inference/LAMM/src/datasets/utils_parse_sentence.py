def parse_sentence(text):
    if '"' in text:
        res = ''
        text_list = text.split('"')
        for i in range(1, len(text_list), 2):
            res += text_list[i] + ' '
        return res[:-1]
    return text
