def clean(self, answer):
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    answer = self.processPunctuation(answer)
    answer = self.processDigitArticle(answer)
    return answer
