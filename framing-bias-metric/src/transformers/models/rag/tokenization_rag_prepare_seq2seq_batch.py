@add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
def prepare_seq2seq_batch(self, src_texts: List[str], tgt_texts: Optional[
    List[str]]=None, max_length: Optional[int]=None, max_target_length:
    Optional[int]=None, padding: str='longest', return_tensors: str=None,
    truncation=True, **kwargs) ->BatchEncoding:
    if max_length is None:
        max_length = self.question_encoder.model_max_length
    model_inputs: BatchEncoding = self.question_encoder(src_texts,
        add_special_tokens=True, return_tensors=return_tensors, max_length=
        max_length, padding=padding, truncation=truncation, **kwargs)
    if tgt_texts is None:
        return model_inputs
    if max_target_length is None:
        max_target_length = self.generator.model_max_length
    labels = self.generator(tgt_texts, add_special_tokens=True,
        return_tensors=return_tensors, padding=padding, max_length=
        max_target_length, truncation=truncation, **kwargs)['input_ids']
    model_inputs['labels'] = labels
    return model_inputs
