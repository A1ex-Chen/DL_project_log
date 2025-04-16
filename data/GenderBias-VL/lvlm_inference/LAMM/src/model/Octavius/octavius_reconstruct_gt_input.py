@torch.no_grad()
def reconstruct_gt_input(self, gt_inputs, task_types):
    gt_inputs = copy.deepcopy(gt_inputs)
    for idx, (gt_input, task_type) in enumerate(zip(gt_inputs, task_types)):
        if task_type == 'detection':
            question, answer = gt_input[0]['value'], gt_input[1]
            bboxes = answer['value']['bboxes']
            classes = answer['value']['clses']
            index = torch.randperm(len(bboxes))
            new_answer = []
            for box_id in index:
                template = np.random.choice(DET_ANSWER_TEMPLATE)
                box_str = (
                    f'{str([round(x, 2) for x in bboxes[box_id].tolist()])}')
                new_answer.append(template.format(P=box_str, C=classes[box_id])
                    )
            new_answer = ' '.join(new_answer)
            gt_inputs[idx][0]['value'] = question
            gt_inputs[idx][1]['value'] = new_answer
    return gt_inputs
