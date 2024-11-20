def setEvalQA(self, quesId, acc):
    self.evalQA[quesId] = round(100 * acc, self.n)
