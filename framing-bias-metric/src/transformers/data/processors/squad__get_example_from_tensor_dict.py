def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
    if not evaluate:
        answer = tensor_dict['answers']['text'][0].numpy().decode('utf-8')
        answer_start = tensor_dict['answers']['answer_start'][0].numpy()
        answers = []
    else:
        answers = [{'answer_start': start.numpy(), 'text': text.numpy().
            decode('utf-8')} for start, text in zip(tensor_dict['answers'][
            'answer_start'], tensor_dict['answers']['text'])]
        answer = None
        answer_start = None
    return SquadExample(qas_id=tensor_dict['id'].numpy().decode('utf-8'),
        question_text=tensor_dict['question'].numpy().decode('utf-8'),
        context_text=tensor_dict['context'].numpy().decode('utf-8'),
        answer_text=answer, start_position_character=answer_start, title=
        tensor_dict['title'].numpy().decode('utf-8'), answers=answers)
