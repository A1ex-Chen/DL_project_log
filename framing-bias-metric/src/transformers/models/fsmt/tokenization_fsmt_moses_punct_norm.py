def moses_punct_norm(self, text, lang):
    if lang not in self.cache_moses_punct_normalizer:
        punct_normalizer = sm.MosesPunctNormalizer(lang=lang)
        self.cache_moses_punct_normalizer[lang] = punct_normalizer
    return self.cache_moses_punct_normalizer[lang].normalize(text)
