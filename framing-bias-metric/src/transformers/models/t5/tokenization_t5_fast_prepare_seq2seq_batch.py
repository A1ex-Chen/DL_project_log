@add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
def prepare_seq2seq_batch(self, src_texts: List[str], tgt_texts: Optional[
    List[str]]=None, max_length: Optional[int]=None, max_target_length:
    Optional[int]=None, padding: str='longest', return_tensors: str=None,
    truncation: bool=True, **kwargs) ->BatchEncoding:
    if max_length is None:
        max_length = self.model_max_length
    self.prefix_tokens = []
    model_inputs = self(src_texts, add_special_tokens=True, return_tensors=
        return_tensors, max_length=max_length, padding=padding, truncation=
        truncation, **kwargs)
    if tgt_texts is None:
        return model_inputs
    if max_target_length is None:
        max_target_length = max_length
    self.prefix_tokens = [self.pad_token_id]
    labels_and_decoder_mask = self(tgt_texts, add_special_tokens=True,
        return_tensors=return_tensors, padding=padding, max_length=
        max_target_length, truncation=truncation, **kwargs)
    model_inputs['labels'] = labels_and_decoder_mask['input_ids']
    self.prefix_tokens = []
    return model_inputs
