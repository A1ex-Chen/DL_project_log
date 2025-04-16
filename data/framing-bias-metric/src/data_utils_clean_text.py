def clean_text(text, remove_stopword=False, remove_nonalphanumeric=False,
    use_number_special_token=False):
    text = text.lower()
    text = re.sub('\\n', ' ', text)
    if remove_nonalphanumeric:
        text = re.sub('([^\\s\\w\\\'.,!?"%]|_)+', ' ', text)
    if use_number_special_token:
        text = re.sub('[-+]?[.\\d]*[\\d]+[:,.\\d]*', '<number>', text)
    if remove_stopword:
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords]
        text = ' '.join(tokens)
    text = re.sub('(http)\\S+', '', text)
    text = re.sub('(www)\\S+', '', text)
    text = re.sub('(href)\\S+', '', text)
    text = re.sub('[ \\s\\t\\n]+', ' ', text)
    return text.strip()
