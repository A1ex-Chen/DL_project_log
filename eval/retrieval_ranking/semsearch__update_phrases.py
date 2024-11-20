def _update_phrases(self):
    if self.extractor:
        self.phrases.clear()
        for sentence in self.sentences:
            for phrase in self.extractor.extract(sentence):
                if phrase.strip().lower() == '':
                    continue
                else:
                    self.phrases.append(phrase.strip().lower())
        self.phrases = list(set(self.phrases))
