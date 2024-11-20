def moses_pipeline(self, text, lang):
    text = replace_unicode_punct(text)
    text = self.moses_punct_norm(text, lang)
    text = remove_non_printing_char(text)
    return text
