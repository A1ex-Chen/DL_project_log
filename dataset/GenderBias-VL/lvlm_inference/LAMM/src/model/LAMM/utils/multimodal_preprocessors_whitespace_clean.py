def whitespace_clean(text):
    text = re.sub('\\s+', ' ', text)
    text = text.strip()
    return text
