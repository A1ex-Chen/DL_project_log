def metric_func(self, answers):
    import re
    CHOICE = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    pattern_1 = re.compile(
        'The answer is \\(?[A-E]\\)?\\W|the answer is \\(?[A-E]\\)?\\W')
    pattern_2 = re.compile('ANSWER: [A-E]')
    pattern_3 = re.compile('\\([A-E]\\)')

    def check_text(text, choices, gt_id):
        text = text.lower()
        if choices[gt_id].lower() not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if choice.lower() in text:
                return False
        return True

    def check_option(res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True

    def check_pattern2(res_list, gt_char):
        pred = res_list[0][-1]
        if pred == gt_char:
            return True
        return False
    score = 0.0
    for item in tqdm(answers, desc='Running Metric'):
        tmp_score = 0
        if 'CoT_answer' in item:
            pred_text = item['CoT_answer'] + item['answer']
        else:
            pred_text = item['answer']
        gt_choice = item['gt_choice']
        gt_char = CHOICE[gt_choice]
        choices = item['choices']
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)
        if len(res_1) != 0:
            if check_option(res_1, gt_char):
                tmp_score = 1.0
        elif len(res_2) != 0:
            if check_pattern2(res_2, gt_char):
                tmp_score = 1.0
        elif len(res_3) != 0:
            if check_option(res_3, gt_char):
                tmp_score = 1.0
        elif check_text(pred_text, choices, gt_choice):
            tmp_score = 1.0
        score += tmp_score
    return dict(vision_acc=score / len(answers)), answers
