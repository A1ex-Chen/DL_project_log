def preprocess(sources: Sequence[str], tokenizer: transformers.
    PreTrainedTokenizer, has_image: bool=False) ->Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '
';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if (conversation_lib.default_conversation.sep_style == conversation_lib
        .SeparatorStyle.PLAIN):
        return preprocess_plain(sources, tokenizer)
    if (conversation_lib.default_conversation.sep_style == conversation_lib
        .SeparatorStyle.LLAMA_2):
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith('v1'):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == 'mpt':
        return preprocess_mpt(sources, tokenizer)
    conversations = []
    for source in sources:
        header = f'{conversation_lib.default_conversation.system}\n\n'
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in
            prompts]
    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer,
            return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized['input_ids']
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s['value'] for s in
                source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s['value'] for s in
                source], tokenizer)['input_ids_lens']
        speakers = [sentence['from'] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)
    return dict(input_ids=input_ids, labels=targets)
