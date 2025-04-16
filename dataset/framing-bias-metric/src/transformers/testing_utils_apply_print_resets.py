def apply_print_resets(buf):
    return re.sub('^.*\\r', '', buf, 0, re.M)
