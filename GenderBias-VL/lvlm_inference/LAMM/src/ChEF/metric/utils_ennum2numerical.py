def ennum2numerical(text):
    text = text.lower()
    for word in text.split():
        if word.isdigit():
            return int(word)
        if word in num_dict:
            return num_dict[word]
    return 0
