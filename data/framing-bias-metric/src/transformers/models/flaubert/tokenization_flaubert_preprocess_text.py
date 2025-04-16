def preprocess_text(self, text):
    text = text.replace('``', '"').replace("''", '"')
    text = convert_to_unicode(text)
    text = unicodedata.normalize('NFC', text)
    if self.do_lowercase:
        text = text.lower()
    return text
