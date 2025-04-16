def decode_generation(pred_ids, input_ids, tokenizer, model_name_or_path):
    pred_ids = [np.where(preds != -100, preds, tokenizer.pad_token_id) for
        preds in pred_ids]
    raw_preds = [tokenizer.batch_decode(preds, skip_special_tokens=True,
        clean_up_tokenization_spaces=True) for preds in pred_ids]
    assert len(raw_preds) == len(input_ids)
    raw_preds = [[pred.strip() for pred in pred_list] for pred_list in
        raw_preds]
    if 'gpt' in model_name_or_path:
        input_ids = [np.where(ipi != -100, ipi, tokenizer.pad_token_id) for
            ipi in input_ids]
        raw_inputs = [tokenizer.batch_decode(ipi, skip_special_tokens=True,
            clean_up_tokenization_spaces=True) for ipi in input_ids]
        raw_inputs = [inp[0].strip() for inp in raw_inputs]
        raw_preds = [[pd[len(inp):].strip() for pd in pd_list] for pd_list,
            inp in zip(raw_preds, raw_inputs)]
    return raw_preds
