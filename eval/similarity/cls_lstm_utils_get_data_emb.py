def get_data_emb(full_run_mode, task, split, model_path, device, shuffle=
    True, contextual=False):
    if task == 'phrase_similarity':
        dataset_path = 'PiC/phrase_similarity'
        data_list = load_dataset(dataset_path)[split]
    else:
        print('Task {} is currently not supported.'.format(task))
        return
    phrase1_list = [item['phrase1'] for item in data_list]
    phrase2_list = [item['phrase2'] for item in data_list]
    labels = [item['label'] for item in data_list]
    context1_list = [item['sentence1'] for item in data_list
        ] if contextual else []
    context2_list = [item['sentence2'] for item in data_list
        ] if contextual else []
    if not full_run_mode:
        subset_size = 50
        phrase1_list = phrase1_list[:subset_size]
        phrase2_list = phrase2_list[:subset_size]
        labels = labels[:subset_size]
    model = load_model(model_path, device)
    if isinstance(model, DensePhrases) or isinstance(model, SimCSE):
        model.model.to(device)
    elif isinstance(model, SentenceTransformer):
        model.to(device)
    print(device)
    emb_batch_size = 32
    if not contextual:
        phrase1_emb_tensor_list = encode_in_batch(model, emb_batch_size,
            phrase1_list, device)
        phrase2_emb_tensor_list = encode_in_batch(model, emb_batch_size,
            phrase2_list, device)
    else:
        phrase1_emb_tensor_list = encode_with_context_in_batch(model,
            emb_batch_size, phrase1_list, context1_list, device)
        phrase2_emb_tensor_list = encode_with_context_in_batch(model,
            emb_batch_size, phrase2_list, context2_list, device)
    combined_phrase_list = []
    for phrase1_emb_tensor, phrase2_emb_tensor, label in zip(
        phrase1_emb_tensor_list, phrase2_emb_tensor_list, labels):
        if phrase1_emb_tensor.shape[0] > 0 and phrase2_emb_tensor.shape[0] > 0:
            combined_phrase_list.append((phrase1_emb_tensor,
                phrase2_emb_tensor, label))
    phrase1_emb_tensor_list, phrase2_emb_tensor_list, labels = zip(*
        combined_phrase_list)
    assert len(phrase1_emb_tensor_list) == len(phrase2_emb_tensor_list)
    if shuffle:
        import random
        random.seed(42)
        combined = list(zip(phrase1_emb_tensor_list,
            phrase2_emb_tensor_list, labels))
        random.shuffle(combined)
        phrase1_emb_tensor_list, phrase2_emb_tensor_list, labels = zip(*
            combined)
    label_tensor = torch.FloatTensor(labels)
    return torch.stack(phrase1_emb_tensor_list), torch.stack(
        phrase2_emb_tensor_list), label_tensor
