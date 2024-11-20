def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for i, c in enumerate(text):
            if c == ' ':
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = ''.join(ns_chars)
        return ns_text, ns_to_s_map
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = ' '.join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text,
                orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1
    orig_ns_text, orig_ns_to_s_map = _strip_spaces(orig_text)
    tok_ns_text, tok_ns_to_s_map = _strip_spaces(tok_text)
    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'"
                , orig_ns_text, tok_ns_text)
        return orig_text
    tok_s_to_ns_map = {}
    for i, tok_index in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i
    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]
    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text
    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]
    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text
    output_text = orig_text[orig_start_position:orig_end_position + 1]
    return output_text
