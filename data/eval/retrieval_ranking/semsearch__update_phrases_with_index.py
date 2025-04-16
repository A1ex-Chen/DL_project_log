def _update_phrases_with_index(self):
    if self.extractor:
        self.phrases.clear()
        for sent_idx, sent in enumerate(self.sentences):
            for phrase, index in self.extractor.extract_with_index(sent):
                if phrase.strip() == '':
                    continue
                self.phrases.append((phrase.strip().lower(), index, sent_idx))
