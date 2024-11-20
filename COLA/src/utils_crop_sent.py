def crop_sent(s, spacy_model, sent_idx=1, offset=3):
    s = list(spacy_model(s).sents)[sent_idx][offset:].text
    s = s.strip(string.punctuation).replace('\n', ' ').strip() + '.'
    return s[0].upper() + s[1:]
