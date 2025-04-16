def transform_sents_to_one_body(body_list):
    body_sent_list = [obj['sentence'] for obj in body_list]
    return ' '.join(body_sent_list)
