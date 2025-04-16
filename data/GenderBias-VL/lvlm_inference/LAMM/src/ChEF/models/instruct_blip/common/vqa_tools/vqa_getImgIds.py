def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
    """
         Get image ids that satisfy given filter conditions. default skips that filter
         :param quesIds   (int array)   : get image ids for given question ids
        quesTypes (str array)   : get image ids for given question types
        ansTypes  (str array)   : get image ids for given answer types
         :return: ids     (int array)   : integer array of image ids
        """
    quesIds = quesIds if type(quesIds) == list else [quesIds]
    quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
    ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]
    if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
        anns = self.dataset['annotations']
    else:
        if not len(quesIds) == 0:
            anns = sum([self.qa[quesId] for quesId in quesIds if quesId in
                self.qa], [])
        else:
            anns = self.dataset['annotations']
        anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann
            ['question_type'] in quesTypes]
        anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann[
            'answer_type'] in ansTypes]
    ids = [ann['image_id'] for ann in anns]
    return ids
