def get_misinfo_datset(args, task, tokenizer, phase='train'):
    if task == 'liar':
        processor = LiarProcessor(args)
    elif task == 'webis':
        processor = WebisProcessor(args)
    elif task == 'clickbait':
        processor = ClickbaitProcessor(args)
    elif task == 'basil_detection':
        processor = BasilBiasDetectionProcessor(args)
    elif task == 'basil_type':
        processor = BasilBiasTypeProcessor(args)
    elif task == 'basil_polarity':
        processor = BasilPolarityProcessor(args)
    elif task == 'fever':
        processor = FeverProcessor(args)
    elif task == 'fever_binary':
        processor = FeverBinaryProcessor(args)
    elif task == 'rumour_detection':
        processor = RumourDetectionProcessor(args)
    elif task == 'rumour_veracity':
        processor = RumourVeracityProcessor(args)
    elif task == 'rumour_veracity_binary':
        processor = RumourVeracityBinaryProcessor(args)
    elif task == 'fnn_politifact' or task == 'fnn_politifact_title':
        processor = FakeNewsNetPolitifactProcessor(args)
    elif task == 'fnn_buzzfeed' or task == 'fnn_buzzfeed_title':
        processor = FakeNewsNetBuzzFeedProcessor(args)
    elif task == 'fnn_gossip':
        processor = FakeNewsNetGossipProcessor(args)
    elif task == 'propaganda':
        processor = PropagandaProcessor(args)
    elif task == 'newstrust':
        pass
    elif task == 'covid_twitter_q1':
        processor = CovidTwitter_Q1_Processor(args)
    elif task == 'covid_twitter_q2':
        processor = CovidTwitter_Q2_Processor(args)
    elif task == 'covid_twitter_q6':
        processor = CovidTwitter_Q6_Processor(args)
    elif task == 'covid_twitter_q7':
        processor = CovidTwitter_Q7_Processor(args)
    else:
        print('wrong task given: {}'.format(task))
        exit(1)
    if phase == 'train':
        examples = processor.get_train_examples()
    elif phase == 'dev':
        examples = processor.get_dev_examples()
    else:
        examples = processor.get_test_examples()
        examples = examples
    features = convert_misinfo_examples_to_features(examples, tokenizer,
        remove_stopwords=args.remove_stopwords, label_map=LABELS[task],
        max_length=args.max_seq_length, pad_on_left=bool(args.model_type in
        ['xlnet']), pad_token=tokenizer.convert_tokens_to_ids([tokenizer.
        pad_token])[0], pad_token_segment_id=4 if args.model_type in [
        'xlnet'] else 0)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=
        torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features],
        dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
        dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    task_idx = torch.tensor([task2idx[task] for _ in features], dtype=torch
        .long)
    all_guids = torch.tensor([f.guid for f in features])
    dataset = TensorDataset(all_input_ids, all_attention_mask,
        all_token_type_ids, all_labels, task_idx, all_guids)
    return dataset
