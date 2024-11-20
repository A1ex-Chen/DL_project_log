def mark_words(text, phrases):
    marked_words = []
    words = re.findall('\\b\\w+\\b|[.,;?!:()"“”‘’\\\']', text)
    word_indices = [match.start() for match in re.finditer(
        '\\b\\w+\\b|[.,;?!:()"“”‘’\\\']', text)]
    for i, word in enumerate(words):
        if any(start <= word_indices[i] < end for _, start, end in phrases):
            marked_words.append((word, 'box'))
        else:
            marked_words.append((word, None))
    return marked_words
