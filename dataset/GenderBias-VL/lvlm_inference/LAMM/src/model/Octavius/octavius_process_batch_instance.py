def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len,
    vision_type, template=conversations.default_conversation):
    batch_input_ids, batch_target_ids = [], []
    for i, conversation in enumerate(batch_of_conversations):
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer,
            conversation, vision_type=vision_type[i], template=template)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True,
        padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True,
        padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()
