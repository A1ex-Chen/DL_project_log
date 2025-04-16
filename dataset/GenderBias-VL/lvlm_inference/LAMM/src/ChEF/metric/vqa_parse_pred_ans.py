def parse_pred_ans(self, pred_ans):
    pred_ans = pred_ans.lower()
    pred_label = None
    if pred_ans in ['yes', 'no']:
        pred_label = pred_ans
    else:
        pred_ans = self.cleaner.clean(pred_ans)
        options = ['yes', 'no']
        answers = []
        for option in options:
            if option in pred_ans:
                answers.append(option)
        if len(answers) != 1:
            pred_label = 'other'
        else:
            pred_label = answers[0]
    return pred_label
