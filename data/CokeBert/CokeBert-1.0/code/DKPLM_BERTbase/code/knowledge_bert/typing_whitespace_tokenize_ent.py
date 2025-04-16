def whitespace_tokenize_ent(text, ents):
    if not text:
        return []
    dd = {}
    for ent in ents:
        dd[ent[0]] = ent[1], ent[2]
    begin = 0
    tokens = []
    entities = []
    while begin < len(text) and text[begin] == ' ':
        begin += 1
    pos = text.find(' ', begin)
    while pos != -1:
        if text[begin:pos] != ' ':
            entity = 'UNK'
            for k, v in dd.items():
                if begin >= v[0] and begin < v[1]:
                    entity = k
                    break
            tokens.append(text[begin:pos])
            entities.append(entity)
        begin = pos
        while begin < len(text) and text[begin] == ' ':
            begin += 1
        pos = text.find(' ', begin)
    if text[begin:] != ' ':
        entity = 'UNK'
        for k, v in dd.items():
            if begin >= v[0] and begin < v[1]:
                entity = k
                break
        tokens.append(text[begin:])
        entities.append(entity)
    return zip(tokens, entities)
