def setEvalQuesType(self, quesId, quesType, acc):
    if quesType not in self.evalQuesType:
        self.evalQuesType[quesType] = {}
    self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)
