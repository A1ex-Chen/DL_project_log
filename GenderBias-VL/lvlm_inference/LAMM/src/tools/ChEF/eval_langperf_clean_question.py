def clean_question(question):
    qlist = question.split('Options:')
    q = qlist[0].split('Context:')
    return 'Question: ' + q[0] + 'Options:' + qlist[1] + '\n'
