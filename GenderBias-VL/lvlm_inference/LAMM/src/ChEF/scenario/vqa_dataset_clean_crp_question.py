def clean_crp_question(self, question):
    qlist = question.split('Options:')
    q = qlist[0].split('Context:')
    q[0] = self.mix_perb.perturb(q[0], self.random_generator)
    return 'Question: ' + q[0] + 'Options:' + qlist[1] + '\n'
