def _lemmatize(self, answers):

    def apply(answer):
        doc = self.lemmatizer(answer)
        words = []
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB']:
                words.append(token.lemma_)
            else:
                words.append(token.text)
        answer = ' '.join(words)
        return answer
    return [apply(answer) for answer in answers]
