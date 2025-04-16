def infer_text(self, answer, choices):
    answer = answer.lower()
    assert isinstance(choices, list)
    gt_choices = {}
    for idx, k in enumerate(choices):
        gt_choices[self.choices[idx]] = str(k).lower()
    cands = []
    for key, value in gt_choices.items():
        if value in answer:
            cands.append(key)
    if len(cands) == 1:
        return cands[0]
    return None
