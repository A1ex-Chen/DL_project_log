def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(
        '(-+|~+|!+|"+|;+|\\?+|\\++|,+|\\)+|\\(+|\\\\+|\\/+|\\*+|\\[+|\\]+|}+|{+|\\|+|_+)'
        , ' \\1 ', text)
    text = re.sub('\\s*\\n\\s*', ' \n ', text)
    text = re.sub('[^\\S\\n]+', ' ', text)
    return text.strip()
