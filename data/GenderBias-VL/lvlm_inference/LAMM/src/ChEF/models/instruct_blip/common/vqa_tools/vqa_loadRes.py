def loadRes(self, resFile, quesFile):
    """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
    res = VQA()
    res.questions = json.load(open(quesFile))
    res.dataset['info'] = copy.deepcopy(self.questions['info'])
    res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
    res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
    res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
    res.dataset['license'] = copy.deepcopy(self.questions['license'])
    print('Loading and preparing results...     ')
    time_t = datetime.datetime.utcnow()
    anns = json.load(open(resFile))
    assert type(anns) == list, 'results is not an array of objects'
    annsQuesIds = [ann['question_id'] for ann in anns]
    assert set(annsQuesIds) == set(self.getQuesIds()
        ), 'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file.'
    for ann in anns:
        quesId = ann['question_id']
        if res.dataset['task_type'] == 'Multiple Choice':
            assert ann['answer'] in self.qqa[quesId]['multiple_choices'
                ], 'predicted answer is not one of the multiple choices'
        qaAnn = self.qa[quesId]
        ann['image_id'] = qaAnn['image_id']
        ann['question_type'] = qaAnn['question_type']
        ann['answer_type'] = qaAnn['answer_type']
    print('DONE (t=%0.2fs)' % (datetime.datetime.utcnow() - time_t).
        total_seconds())
    res.dataset['annotations'] = anns
    res.createIndex()
    return res
