def remove_articles(text):
    regex = re.compile('\\b(a|an|the)\\b', re.UNICODE)
    return re.sub(regex, ' ', text)
