def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()
