def setEvalAnsType(self, quesId, ansType, acc):
    if ansType not in self.evalAnsType:
        self.evalAnsType[ansType] = {}
    self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)
