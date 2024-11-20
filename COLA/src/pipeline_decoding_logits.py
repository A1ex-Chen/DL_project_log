def decoding_logits(self, preds_tuple):
    topk_values, topk_indices = preds_tuple
    pred_list = []
    for value_list, indices_list in tqdm(zip(topk_values, topk_indices),
        'decoding tokens', total=len(topk_indices)):
        cur_pred_list = []
        for v, i in zip(value_list, indices_list):
            cur_pred_list.append({'token_str': self._tokenizer.decode([i]),
                'score': v})
        pred_list.append(cur_pred_list)
    return pred_list
