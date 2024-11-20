def get_tag(tokenized, tags):
    if not isinstance(tags, (list, tuple)):
        tags = [tags]
    ret = []
    for word, pos in nltk.pos_tag(tokenized):
        for tag in tags:
            if pos == tag:
                ret.append(word)
    return ret
