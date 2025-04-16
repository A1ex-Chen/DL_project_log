def eval_sub_data(self, sub_data, answer_map):
    lt = len(sub_data)
    GT, PRED = [], []
    result = 1
    for i in range(lt):
        item = sub_data[i]
        idx = item['id']
        GT.append(self.choices[answer_map[idx]])
        pred_answer, option_match, content_match = (self.answer_extractor.
            fetch_answer(item['answer'], item['choices']))
        PRED.append(pred_answer)
        if pred_answer is not None:
            self.match_content += content_match
            self.match_option += option_match
            if GT[-1] != PRED[-1]:
                result = 0
        else:
            result = 0
    return result
