def moses_detokenize(self, tokens, lang):
    if lang not in self.cache_moses_tokenizer:
        moses_detokenizer = sm.MosesDetokenizer(lang=self.tgt_lang)
        self.cache_moses_detokenizer[lang] = moses_detokenizer
    return self.cache_moses_detokenizer[lang].detokenize(tokens)
