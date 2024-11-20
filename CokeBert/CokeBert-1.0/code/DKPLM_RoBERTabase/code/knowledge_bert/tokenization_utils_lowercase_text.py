def lowercase_text(t):
    escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
    pattern = '(' + '|'.join(escaped_special_toks) + ')|' + '(.+?)'
    return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)
