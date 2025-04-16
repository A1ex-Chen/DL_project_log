@add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
def prepare_seq2seq_batch(self, src_texts: List[str], src_lang: str='en_XX',
    tgt_texts: Optional[List[str]]=None, tgt_lang: str='ro_RO', max_length:
    Optional[int]=None, max_target_length: Optional[int]=None, truncation:
    bool=True, padding: str='longest', return_tensors: str=None, **kwargs
    ) ->BatchEncoding:
    if max_length is None:
        max_length = self.model_max_length
    self.set_src_lang_special_tokens(src_lang)
    model_inputs: BatchEncoding = self(src_texts, add_special_tokens=True,
        return_tensors=return_tensors, max_length=max_length, padding=
        padding, truncation=truncation, **kwargs)
    if tgt_texts is None:
        return model_inputs
    if max_target_length is None:
        max_target_length = max_length
    self.set_tgt_lang_special_tokens(tgt_lang)
    labels = self(tgt_texts, add_special_tokens=True, return_tensors=
        return_tensors, padding=padding, max_length=max_target_length,
        truncation=True, **kwargs)['input_ids']
    model_inputs['labels'] = labels
    self.set_src_lang_special_tokens(src_lang)
    return model_inputs
