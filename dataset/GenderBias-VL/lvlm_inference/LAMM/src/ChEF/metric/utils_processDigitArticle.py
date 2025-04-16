def processDigitArticle(self, inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = self.manualMap.setdefault(word, word)
        if word not in self.articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in self.contractions:
            outText[wordId] = self.contractions[word]
    outText = ' '.join(outText)
    return outText
