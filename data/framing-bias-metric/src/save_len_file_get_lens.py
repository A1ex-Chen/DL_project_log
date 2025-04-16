def get_lens(ds):
    dl = tqdm(DataLoader(ds, batch_size=512, num_workers=8, shuffle=False,
        collate_fn=ds.collate_fn), desc=str(ds.len_file))
    max_lens = []
    for batch in dl:
        src_lens = batch['input_ids'].ne(pad).sum(1).tolist()
        tgt_lens = batch['labels'].ne(pad).sum(1).tolist()
        if consider_target:
            for src, tgt in zip(src_lens, tgt_lens):
                max_lens.append(max(src, tgt))
        else:
            max_lens.extend(src_lens)
    return max_lens
