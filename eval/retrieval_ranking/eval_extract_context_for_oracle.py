def extract_context_for_oracle(sentences, answer):
    gt_sentence = ''
    gt_sentence_idx = -1
    for idx, sentence in enumerate(sentences):
        if answer.strip().lower() in sentence.strip().lower():
            gt_sentence = sentence
            gt_sentence_idx = idx
    return gt_sentence, gt_sentence_idx
