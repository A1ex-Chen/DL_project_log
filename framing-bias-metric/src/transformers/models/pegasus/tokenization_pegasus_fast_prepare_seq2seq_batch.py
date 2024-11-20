@add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
def prepare_seq2seq_batch(self, src_texts: List[str], tgt_texts: Optional[
    List[str]]=None, max_length: Optional[int]=None, max_target_length:
    Optional[int]=None, return_tensors: str=None, truncation=True, padding=
    'longest', **unused) ->BatchEncoding:
    if '' in src_texts:
        raise ValueError(f'found empty string in src_texts: {src_texts}')
    tokenizer_kwargs = dict(add_special_tokens=True, return_tensors=
        return_tensors, max_length=max_length, truncation=truncation,
        padding=padding)
    model_inputs: BatchEncoding = self(src_texts, **tokenizer_kwargs)
    if tgt_texts is None:
        return model_inputs
    if max_target_length is not None:
        tokenizer_kwargs['max_length'] = max_target_length
    labels: BatchEncoding = self(tgt_texts, **tokenizer_kwargs)['input_ids']
    model_inputs['labels'] = labels
    return model_inputs
