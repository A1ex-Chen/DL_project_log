def pre_question(self, question):
    question = re.sub('([.!\\"()*#:;~])', '', question.lower())
    question = question.rstrip(' ')
    question_words = question.split(' ')
    if len(question_words) > self.max_words:
        question = ' '.join(question_words[:self.max_words])
    return question
