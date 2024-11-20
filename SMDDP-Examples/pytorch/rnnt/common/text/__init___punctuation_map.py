def punctuation_map(labels):
    punctuation = string.punctuation
    punctuation = punctuation.replace('+', '')
    punctuation = punctuation.replace('&', '')
    for l in labels:
        punctuation = punctuation.replace(l, '')
    table = str.maketrans(punctuation, ' ' * len(punctuation))
    return table
