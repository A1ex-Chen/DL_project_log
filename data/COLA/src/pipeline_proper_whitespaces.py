def proper_whitespaces(text):
    return re.sub('\\s([?.!"](?:\\s|$))', '\\1', text.strip())
