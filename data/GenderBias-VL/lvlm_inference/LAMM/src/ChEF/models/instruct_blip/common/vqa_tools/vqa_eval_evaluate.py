def evaluate(self, quesIds=None):
    if quesIds == None:
        quesIds = [quesId for quesId in self.params['question_id']]
    gts = {}
    res = {}
    for quesId in quesIds:
        gts[quesId] = self.vqa.qa[quesId]
        res[quesId] = self.vqaRes.qa[quesId]
    accQA = []
    accQuesType = {}
    accAnsType = {}
    print('computing accuracy')
    step = 0
    for quesId in quesIds:
        resAns = res[quesId]['answer']
        resAns = resAns.replace('\n', ' ')
        resAns = resAns.replace('\t', ' ')
        resAns = resAns.strip()
        resAns = self.processPunctuation(resAns)
        resAns = self.processDigitArticle(resAns)
        gtAcc = []
        gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
        if len(set(gtAnswers)) > 1:
            for ansDic in gts[quesId]['answers']:
                ansDic['answer'] = self.processPunctuation(ansDic['answer'])
        for gtAnsDatum in gts[quesId]['answers']:
            otherGTAns = [item for item in gts[quesId]['answers'] if item !=
                gtAnsDatum]
            matchingAns = [item for item in otherGTAns if item['answer'] ==
                resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        quesType = gts[quesId]['question_type']
        ansType = gts[quesId]['answer_type']
        avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
        accQA.append(avgGTAcc)
        if quesType not in accQuesType:
            accQuesType[quesType] = []
        accQuesType[quesType].append(avgGTAcc)
        if ansType not in accAnsType:
            accAnsType[ansType] = []
        accAnsType[ansType].append(avgGTAcc)
        self.setEvalQA(quesId, avgGTAcc)
        self.setEvalQuesType(quesId, quesType, avgGTAcc)
        self.setEvalAnsType(quesId, ansType, avgGTAcc)
        if step % 100 == 0:
            self.updateProgress(step / float(len(quesIds)))
        step = step + 1
    self.setAccuracy(accQA, accQuesType, accAnsType)
    print('Done computing accuracy')
