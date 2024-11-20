def setAccuracy(self, accQA, accQuesType, accAnsType):
    self.accuracy['overall'] = round(100 * float(sum(accQA)) / len(accQA),
        self.n)
    self.accuracy['perQuestionType'] = {quesType: round(100 * float(sum(
        accQuesType[quesType])) / len(accQuesType[quesType]), self.n) for
        quesType in accQuesType}
    self.accuracy['perAnswerType'] = {ansType: round(100 * float(sum(
        accAnsType[ansType])) / len(accAnsType[ansType]), self.n) for
        ansType in accAnsType}
