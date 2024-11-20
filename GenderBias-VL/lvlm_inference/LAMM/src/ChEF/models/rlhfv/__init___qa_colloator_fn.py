def qa_colloator_fn(data_list, tokenizer, img_transform):
    questions = [x['question'] for x in data_list]
    tokenized = tokenizer(questions)
    input_ids = [torch.as_tensor(v) for v in tokenized['input_ids']]
    input_ids = torch_pad_sequence(input_ids, tokenizer.pad_token_id,
        padding_side='left')
    attn_mask = [torch.as_tensor(v) for v in tokenized['attention_mask']]
    attn_mask = torch_pad_sequence(attn_mask, 0, padding_side='left')
    images = [img_transform(x['image']) for x in data_list]
    images = torch.stack(images)
    raw_questions = [x['raw_question'] for x in data_list]
    data = {'images': images, 'input_ids': input_ids, 'attention_mask':
        attn_mask, 'raw_questions': raw_questions}
    if 'question_id' in data_list[0]:
        data['question_id'] = [x['question_id'] for x in data_list]
    return data
