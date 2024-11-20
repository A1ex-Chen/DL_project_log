def remove_blanks(text):
    try:
        before, answers = text.split(self.SEP_TOK)
        answers = [x.strip() for x in answers.split(self.ANSWER_TOK)][:-1]
        answers = [(x if x != self.EMPTY_TOK else '') for x in answers]
        for a in answers:
            if a == '':
                before = re.sub(' %s' % re.escape(self.BLANK_TOK), a,
                    before, count=1)
            else:
                before = re.sub('%s' % re.escape(self.BLANK_TOK), a, before,
                    count=1)
        return before, answers
    except:
        return text, []
