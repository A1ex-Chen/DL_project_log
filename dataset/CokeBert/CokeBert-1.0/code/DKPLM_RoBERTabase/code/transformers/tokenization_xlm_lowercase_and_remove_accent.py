def lowercase_and_remove_accent(text):
    """
    Lowercase and strips accents from a piece of text based on
    https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    text = ' '.join(text)
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == 'Mn':
            continue
        output.append(char)
    return ''.join(output).lower().split(' ')
