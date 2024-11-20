def moses_tokenize(self, text):
    return self.moses_tokenizer.tokenize(text, aggressive_dash_splits=True,
        return_str=False, escape=False, protected_patterns=self.never_split)
