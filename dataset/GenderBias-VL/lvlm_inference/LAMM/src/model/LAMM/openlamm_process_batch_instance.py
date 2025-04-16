def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len,
    vision_type='image', template=conversations.default_conversation):
    """build one batch of instance for training

    :param class tokenizer: text tokenizer
    :param list batch_of_conversations: batch of conversations
    :param int max_tgt_len: max token length of after vision tokens
    :param str vision_type: type of vision data, defaults to 'image'
    :return list: input token ids, target token ids, attention mask
    """
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer,
            conversation, vision_type=vision_type, template=template)
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
