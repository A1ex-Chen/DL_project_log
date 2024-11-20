def tokenize(self, text):
    """
        Args:
            text: str

        Returns: list(str) A tokenized list of strings; concatenating this list returns the original string if
        `preserve_case=False`
        """
    text = _replace_html_entities(text)
    if self.strip_handles:
        text = remove_handles(text)
    if self.reduce_len:
        text = reduce_lengthening(text)
    safe_text = HANG_RE.sub('\\1\\1\\1', text)
    words = WORD_RE.findall(safe_text)
    if not self.preserve_case:
        words = list(map(lambda x: x if EMOTICON_RE.search(x) else x.lower(
            ), words))
    return words
