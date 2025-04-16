def remove_punctuation(text, table):
    text = text.translate(table)
    text = re.sub('&', ' and ', text)
    text = re.sub('\\+', ' plus ', text)
    return text
