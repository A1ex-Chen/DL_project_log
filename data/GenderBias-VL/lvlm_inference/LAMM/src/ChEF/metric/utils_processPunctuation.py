def processPunctuation(self, inText):
    outText = inText
    for p in self.punct:
        if p + ' ' in inText or ' ' + p in inText:
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = self.periodStrip.sub('', outText, re.UNICODE)
    return outText
