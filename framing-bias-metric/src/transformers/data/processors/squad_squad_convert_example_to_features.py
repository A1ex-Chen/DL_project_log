def squad_convert_example_to_features(example, max_seq_length, doc_stride,
    max_query_length, padding_strategy, is_training):
    features = []
    if is_training and not example.is_impossible:
        start_position = example.start_position
        end_position = example.end_position
        actual_text = ' '.join(example.doc_tokens[start_position:
            end_position + 1])
        cleaned_answer_text = ' '.join(whitespace_tokenize(example.answer_text)
            )
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'",
                actual_text, cleaned_answer_text)
            return []
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in ['RobertaTokenizer',
            'LongformerTokenizer', 'BartTokenizer', 'RobertaTokenizerFast',
            'LongformerTokenizerFast', 'BartTokenizerFast']:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        tok_start_position, tok_end_position = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.answer_text)
    spans = []
    truncated_query = tokenizer.encode(example.question_text,
        add_special_tokens=False, truncation=True, max_length=max_query_length)
    tokenizer_type = type(tokenizer).__name__.replace('Tokenizer', '').lower()
    sequence_added_tokens = (tokenizer.model_max_length - tokenizer.
        max_len_single_sentence + 1 if tokenizer_type in
        MULTI_SEP_TOKENS_TOKENIZERS_SET else tokenizer.model_max_length -
        tokenizer.max_len_single_sentence)
    sequence_pair_added_tokens = (tokenizer.model_max_length - tokenizer.
        max_len_sentences_pair)
    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        if tokenizer.padding_side == 'right':
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value
        encoded_dict = tokenizer.encode_plus(texts, pairs, truncation=
            truncation, padding=padding_strategy, max_length=max_seq_length,
            return_overflowing_tokens=True, stride=max_seq_length -
            doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True)
        paragraph_len = min(len(all_doc_tokens) - len(spans) * doc_stride, 
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens)
        if tokenizer.pad_token_id in encoded_dict['input_ids']:
            if tokenizer.padding_side == 'right':
                non_padded_ids = encoded_dict['input_ids'][:encoded_dict[
                    'input_ids'].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = len(encoded_dict['input_ids']
                    ) - 1 - encoded_dict['input_ids'][::-1].index(tokenizer
                    .pad_token_id)
                non_padded_ids = encoded_dict['input_ids'][
                    last_padding_id_position + 1:]
        else:
            non_padded_ids = encoded_dict['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = (len(truncated_query) + sequence_added_tokens + i if 
                tokenizer.padding_side == 'right' else i)
            token_to_orig_map[index] = tok_to_orig_index[len(spans) *
                doc_stride + i]
        encoded_dict['paragraph_len'] = paragraph_len
        encoded_dict['tokens'] = tokens
        encoded_dict['token_to_orig_map'] = token_to_orig_map
        encoded_dict['truncated_query_with_special_tokens_length'] = len(
            truncated_query) + sequence_added_tokens
        encoded_dict['token_is_max_context'] = {}
        encoded_dict['start'] = len(spans) * doc_stride
        encoded_dict['length'] = paragraph_len
        spans.append(encoded_dict)
        if ('overflowing_tokens' not in encoded_dict or 
            'overflowing_tokens' in encoded_dict and len(encoded_dict[
            'overflowing_tokens']) == 0):
            break
        span_doc_tokens = encoded_dict['overflowing_tokens']
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]['paragraph_len']):
            is_max_context = _new_check_is_max_context(spans,
                doc_span_index, doc_span_index * doc_stride + j)
            index = j if tokenizer.padding_side == 'left' else spans[
                doc_span_index]['truncated_query_with_special_tokens_length'
                ] + j
            spans[doc_span_index]['token_is_max_context'][index
                ] = is_max_context
    for span in spans:
        cls_index = span['input_ids'].index(tokenizer.cls_token_id)
        p_mask = np.ones_like(span['token_type_ids'])
        if tokenizer.padding_side == 'right':
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span['tokens']):-(len(truncated_query) +
                sequence_added_tokens)] = 0
        pad_token_indices = np.where(span['input_ids'] == tokenizer.
            pad_token_id)
        special_token_indices = np.asarray(tokenizer.
            get_special_tokens_mask(span['input_ids'],
            already_has_special_tokens=True)).nonzero()
        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1
        p_mask[cls_index] = 0
        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            doc_start = span['start']
            doc_end = span['start'] + span['length'] - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and tok_end_position <=
                doc_end):
                out_of_span = True
            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == 'left':
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        features.append(SquadFeatures(span['input_ids'], span[
            'attention_mask'], span['token_type_ids'], cls_index, p_mask.
            tolist(), example_index=0, unique_id=0, paragraph_len=span[
            'paragraph_len'], token_is_max_context=span[
            'token_is_max_context'], tokens=span['tokens'],
            token_to_orig_map=span['token_to_orig_map'], start_position=
            start_position, end_position=end_position, is_impossible=
            span_is_impossible, qas_id=example.qas_id))
    return features
