def tokenize(_list):
    new_dict = {}
    for item in _list:
        if isinstance(item, list):
            new_sentence_list = []
            for sentence in item:
                a = ''
                for token in nlp(sentence):
                    a += token.text
                    a += ' '
                new_sentence_list.append(a.rstrip())
            new_dict[len(new_dict)] = new_sentence_list
        else:
            a = ''
            for token in nlp(item):
                a += token.text
                a += ' '
            new_dict[len(new_dict)] = [a]
    return new_dict
