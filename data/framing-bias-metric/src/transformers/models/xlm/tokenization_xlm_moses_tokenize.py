def moses_tokenize(self, text, lang):
    if lang not in self.cache_moses_tokenizer:
        moses_tokenizer = sm.MosesTokenizer(lang=lang)
        self.cache_moses_tokenizer[lang] = moses_tokenizer
    else:
        moses_tokenizer = self.cache_moses_tokenizer[lang]
    return moses_tokenizer.tokenize(text, return_str=False, escape=False)
