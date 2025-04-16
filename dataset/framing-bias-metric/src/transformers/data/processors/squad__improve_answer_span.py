def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
    orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = ' '.join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = ' '.join(doc_tokens[new_start:new_end + 1])
            if text_span == tok_answer_text:
                return new_start, new_end
    return input_start, input_end
