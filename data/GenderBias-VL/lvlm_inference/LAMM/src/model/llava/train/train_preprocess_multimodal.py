def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments
    ) ->Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence[
                    'value']
                sentence['value'] = sentence['value'].strip()
                if 'mmtag' in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' +
                        DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (DEFAULT_IM_START_TOKEN + replace_token +
                    DEFAULT_IM_END_TOKEN)
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                replace_token)
    return sources
