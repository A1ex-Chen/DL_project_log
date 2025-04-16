def showQA(self, anns):
    """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
    if len(anns) == 0:
        return 0
    for ann in anns:
        quesId = ann['question_id']
        print('Question: %s' % self.qqa[quesId]['question'])
        for ans in ann['answers']:
            print('Answer %d: %s' % (ans['answer_id'], ans['answer']))
