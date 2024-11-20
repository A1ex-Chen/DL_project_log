def call_on_point(self, sentence: str, points_seq: BoxesSeq) ->str:
    all_box = self.points_token_pat.findall(sentence)
    assert len(all_box) == len(points_seq
        ), f'not match. sentence: {sentence}. boxes:{points_seq}'
    if len(all_box) == 0:
        return sentence
    bboxes_strs = [self.format_point(bboxes) for bboxes in points_seq]
    converted = sentence.replace(self.points_token, '{}').format(*bboxes_strs)
    return converted
