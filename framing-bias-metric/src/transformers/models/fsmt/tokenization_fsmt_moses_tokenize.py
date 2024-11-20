def moses_tokenize(self, text, lang):
    if lang not in self.cache_moses_tokenizer:
        moses_tokenizer = sm.MosesTokenizer(lang=lang)
        self.cache_moses_tokenizer[lang] = moses_tokenizer
    return self.cache_moses_tokenizer[lang].tokenize(text,
        aggressive_dash_splits=True, return_str=False, escape=True)
