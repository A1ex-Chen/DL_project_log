def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
    """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
    cand_indexes = []
    for i, token in enumerate(input_tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        if len(cand_indexes) >= 1 and token.startswith('##'):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    random.shuffle(cand_indexes)
    num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens
        ) * self.mlm_probability))))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_lms.append(index)
    assert len(covered_indexes) == len(masked_lms)
    mask_labels = [(1 if i in covered_indexes else 0) for i in range(len(
        input_tokens))]
    return mask_labels
