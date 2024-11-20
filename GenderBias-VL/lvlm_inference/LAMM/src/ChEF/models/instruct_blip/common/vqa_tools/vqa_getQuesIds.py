def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
    """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                        quesTypes (str array)   : get question ids for given question types
                        ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
    imgIds = imgIds if type(imgIds) == list else [imgIds]
    quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
    ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]
    if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
        anns = self.dataset['annotations']
    else:
        if not len(imgIds) == 0:
            anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in
                self.imgToQA], [])
        else:
            anns = self.dataset['annotations']
        anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann
            ['question_type'] in quesTypes]
        anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann[
            'answer_type'] in ansTypes]
    ids = [ann['question_id'] for ann in anns]
    return ids
