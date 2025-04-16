def __call__(self, questions, titles: Optional[str]=None, texts: Optional[
    str]=None, padding: Union[bool, str]=False, truncation: Union[bool, str
    ]=False, max_length: Optional[int]=None, return_tensors: Optional[Union
    [str, TensorType]]=None, return_attention_mask: Optional[bool]=None, **
    kwargs) ->BatchEncoding:
    if titles is None and texts is None:
        return super().__call__(questions, padding=padding, truncation=
            truncation, max_length=max_length, return_tensors=
            return_tensors, return_attention_mask=return_attention_mask, **
            kwargs)
    elif titles is None or texts is None:
        text_pair = titles if texts is None else texts
        return super().__call__(questions, text_pair, padding=padding,
            truncation=truncation, max_length=max_length, return_tensors=
            return_tensors, return_attention_mask=return_attention_mask, **
            kwargs)
    titles = titles if not isinstance(titles, str) else [titles]
    texts = texts if not isinstance(texts, str) else [texts]
    n_passages = len(titles)
    questions = questions if not isinstance(questions, str) else [questions
        ] * n_passages
    assert len(titles) == len(texts
        ), 'There should be as many titles than texts but got {} titles and {} texts.'.format(
        len(titles), len(texts))
    encoded_question_and_titles = super().__call__(questions, titles,
        padding=False, truncation=False)['input_ids']
    encoded_texts = super().__call__(texts, add_special_tokens=False,
        padding=False, truncation=False)['input_ids']
    encoded_inputs = {'input_ids': [((encoded_question_and_title +
        encoded_text)[:max_length] if max_length is not None and truncation
         else encoded_question_and_title + encoded_text) for 
        encoded_question_and_title, encoded_text in zip(
        encoded_question_and_titles, encoded_texts)]}
    if return_attention_mask is not False:
        attention_mask = [(input_ids != self.pad_token_id) for input_ids in
            encoded_inputs['input_ids']]
        encoded_inputs['attention_mask'] = attention_mask
    return self.pad(encoded_inputs, padding=padding, max_length=max_length,
        return_tensors=return_tensors)
