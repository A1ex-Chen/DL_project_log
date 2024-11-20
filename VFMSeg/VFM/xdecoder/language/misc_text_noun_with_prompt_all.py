def text_noun_with_prompt_all(text, phrase_prob=0.0, append_text=True):
    tokenized = nltk.word_tokenize(text)
    if random.random() >= phrase_prob:
        nouns = get_tag(tokenized, ['NN', 'NNS', 'NNP'])
    else:
        nouns = get_noun_phrase(tokenized)
    prompt_texts = [np.random.choice(IMAGENET_DEFAULT_TEMPLATES).format(
        noun) for noun in nouns]
    if append_text:
        prompt_texts += [text]
        nouns += [text]
    return prompt_texts, nouns
