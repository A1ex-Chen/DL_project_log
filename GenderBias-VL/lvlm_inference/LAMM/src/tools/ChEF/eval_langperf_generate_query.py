def generate_query(gt_data, answer):
    question = clean_question(gt_data['query'])
    gt_choice = gt_data['gt_choice']
    gt = option_text(gt_choice)
    Cot = answer['CoT_answer']
    idx = Cot.rfind('.')
    if idx > 0:
        Cot = Cot[:idx + 1]
    answer_text = answer['answer'] + ')'
    query = DATA_Template.format(question, gt, Cot, answer_text)
    return query
