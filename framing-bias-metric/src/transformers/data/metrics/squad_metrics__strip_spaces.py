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
