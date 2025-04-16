def load_basil_triples(just_body=False):
    with open(
        '/home/nayeon/omission/emnlp19-media-bias/basil_with_neutral.json', 'r'
        ) as infile:
        news_triples = json.load(infile)
    triples = []
    for triple_idx in news_triples:
        triple = news_triples[str(triple_idx)]
        if just_body:
            if triple['neutral'] != None:
                center = ' '.join(triple['neutral']['body'])
            else:
                center = transform_sents_to_one_body(triple['nyt']['body'])
            left = transform_sents_to_one_body(triple['hpo']['body'])
            right = transform_sents_to_one_body(triple['fox']['body'])
        else:
            if triple['neutral'] != None:
                center = triple['neutral']
            else:
                center = triple['nyt']
            left = triple['hpo']
            right = triple['fox']
        triples += {'id': triple_idx, 'center': center, 'left': left,
            'right': right},
    return triples
