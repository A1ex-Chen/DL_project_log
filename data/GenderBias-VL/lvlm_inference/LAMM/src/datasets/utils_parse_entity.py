def parse_entity(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in string.punctuation]
    words = [word for word in words if word not in stops]
    words = [wordnet.morphy(word) for word in words if word not in stops]
    return words
