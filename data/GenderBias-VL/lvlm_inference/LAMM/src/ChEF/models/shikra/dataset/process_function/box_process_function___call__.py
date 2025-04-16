def __call__(self, sentence: str, bboxes_seq: BoxesSeq) ->str:
    all_box = self.bboxes_token_pat.findall(sentence)
    assert len(all_box) == len(bboxes_seq
        ), f'not match. sentence: {sentence}. boxes:{bboxes_seq}'
    if len(all_box) == 0:
        return sentence
    bboxes_strs = [self.format_box(bboxes) for bboxes in bboxes_seq]
    converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
    return converted
