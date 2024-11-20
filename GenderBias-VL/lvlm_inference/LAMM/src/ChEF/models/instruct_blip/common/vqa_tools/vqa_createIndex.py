def createIndex(self):
    print('creating index...')
    imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
    qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
    qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
    for ann in self.dataset['annotations']:
        imgToQA[ann['image_id']] += [ann]
        qa[ann['question_id']] = ann
    for ques in self.questions['questions']:
        qqa[ques['question_id']] = ques
    print('index created!')
    self.qa = qa
    self.qqa = qqa
    self.imgToQA = imgToQA
