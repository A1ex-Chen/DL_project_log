def clean_question(question, generative=False):
    qlist = question.split('Options:')
    q = qlist[0].split('Context:')
    if not generative:
        res = 'Question: ' + q[0] + 'Options:' + qlist[1] + '\n'
    else:
        res = 'Question: ' + q[0]
    return res
