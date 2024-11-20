def collate_fn(x):
    x = torch.LongTensor(np.array(x))
    entity_idx = x[:, 4 * args.max_seq_length:5 * args.max_seq_length]
    uniq_idx = np.unique(entity_idx.numpy())
    ent_candidate = torch.LongTensor(uniq_idx + 1)
    ent_candidate = ent_candidate.repeat([torch.cuda.device_count(), 1])
    d = {}
    dd = []
    for i, idx in enumerate(uniq_idx):
        d[idx] = i
        dd.append(idx)
    ent_size = len(uniq_idx) - 1

    def map(x):
        if x == -1:
            return -1
        else:
            rnd = random.uniform(0, 1)
            if rnd < 0.05:
                return dd[random.randint(1, ent_size)]
            elif rnd < 0.2:
                return -1
            else:
                return x
    ent_labels = entity_idx.clone()
    d[-1] = -1
    ent_labels = ent_labels.apply_(lambda x: d[x])
    entity_idx.apply_(map)
    ent_emb = entity_idx + 1
    mask = entity_idx.clone()
    mask.apply_(lambda x: 0 if x == -1 else 1)
    mask[:, 0] = 1
    return x[:, :args.max_seq_length], x[:, args.max_seq_length:2 * args.
        max_seq_length], x[:, 2 * args.max_seq_length:3 * args.max_seq_length
        ], x[:, 3 * args.max_seq_length:4 * args.max_seq_length
        ], ent_emb, mask, x[:, 6 * args.max_seq_length:
        ], ent_candidate, ent_labels
