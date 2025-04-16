def find_left_right_intersection(target_leaning_name, basil_triples,
    use_truncation=False):
    selected_intersection = []
    target_leaning = target_leaning_name
    other_leaning = 'left' if target_leaning_name == 'right' else 'right'
    for idx, triple in tqdm(enumerate(basil_triples), total=len(basil_triples)
        ):
        if use_truncation:
            target = truncate(triple[target_leaning])
            other = truncate(triple[other_leaning])
        else:
            target = triple[target_leaning]
            other = triple[other_leaning]
        target_sents = sent_tokenize(target)
        selected = find_match(target_sents, other)
        selected_text = ' '.join(selected)
        selected_intersection.append(selected_text)
    return selected_intersection
