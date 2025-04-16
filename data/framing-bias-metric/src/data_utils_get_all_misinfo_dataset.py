def get_all_misinfo_dataset(args, tokenizer, phase='train'):
    processor = AllCombinedProcessor(args)
    if phase == 'train':
        all_examples = processor.get_train_examples()
    elif phase == 'dev':
        all_examples = processor.get_dev_examples()
    else:
        all_examples = processor.get_test_examples()
        all_examples = all_examples
    (combined_all_input_ids, combined_all_attention_mask,
        combined_all_token_type_ids) = [], [], []
    combined_all_labels, combined_task_idx, combined_all_guids = [], [], []
    for examples in all_examples:
        features = convert_misinfo_examples_to_features(examples, tokenizer,
            remove_stopwords=args.remove_stopwords, label_map=LABELS[task],
            max_length=args.max_seq_length, pad_on_left=bool(args.
            model_type in ['xlnet']), pad_token=tokenizer.
            convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype
            =torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in
            features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in
            features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long
            )
        task_idx = torch.tensor([task2idx[task] for _ in features], dtype=
            torch.long)
        all_guids = torch.tensor([f.guid for f in features])
        combined_all_input_ids.append(all_input_ids)
        combined_all_attention_mask.append(all_attention_mask)
        combined_all_token_type_ids.append(all_token_type_ids)
        combined_all_labels.append(all_labels)
        combined_task_idx.append(task_idx)
        combined_all_guids.append(all_guids)
    dataset = TensorDataset(combined_all_input_ids,
        combined_all_attention_mask, combined_all_all_token_type_ids,
        combined_all_all_labels, combined_all_task_idx, combined_all_all_guids)
    return dataset
