def convert_examples_to_features(examples, label_list, max_seq_length,
    tokenizer_label, tokenizer, threshold):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    entity2id = {}
    with open('../../data/kg_embed/entity2id.txt') as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            entity2id[qid] = int(eid)
    features = []
    for ex_index, example in enumerate(examples):
        ex_text_a = example.text_a[0]
        h = example.text_a[1][0]
        ex_text_a = ex_text_a[:h[1]] + '∞ ' + ex_text_a[h[1]:h[2]
            ] + ' º' + ex_text_a[h[2]:]
        begin, end = h[1:3]
        h[1] += 2
        h[2] += 2
        tokens_a, entities_a = tokenizer_label.tokenize(ex_text_a, [h])
        ent_pos = [x for x in example.text_b if x[-1] > threshold]
        for x in ent_pos:
            if x[1] > end:
                x[1] += 4
            elif x[1] >= begin:
                x[1] += 2
        _, entities = tokenizer.tokenize(ex_text_a, ent_pos)
        if h[1] == h[2]:
            continue
        mark = False
        tokens_b = None
        for e in entities_a:
            if e != 'UNK':
                mark = True
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
            entities_a = entities_a[:max_seq_length - 2]
            entities = entities[:max_seq_length - 2]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        ents = ['UNK'] + entities_a + ['UNK']
        real_ents = ['UNK'] + entities + ['UNK']
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        span_mask = []
        for ent in ents:
            if ent != 'UNK':
                span_mask.append(1)
            else:
                span_mask.append(0)
        input_ent = []
        ent_mask = []
        for ent in real_ents:
            if ent != 'UNK' and ent in entity2id:
                input_ent.append(entity2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1
        if not mark:
            print(example.guid)
            print(example.text_a[0])
            print(example.text_a[0][example.text_a[1][0][1]:example.text_a[
                1][0][2]])
            print(ents)
            exit(1)
        if sum(span_mask) == 0:
            continue
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        ent_mask += padding
        input_ent += padding_
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(ent_mask) == max_seq_length
        assert len(input_ent) == max_seq_length
        labels = [0] * len(label_map)
        for l in example.label:
            l = label_map[l]
            labels[l] = 1
        if ex_index < 0:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('Entity: %s' % example.text_a[1])
            logger.info('tokens: %s' % ' '.join([str(x) for x in zip(tokens,
                ents)]))
            logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids])
                )
            logger.info('label: %s %s' % (example.label, labels))
            logger.info(real_ents)
        f = 0
        for i in input_ent:
            if i != 0:
                break
            else:
                f = 1
        if f != 0:
            print(example.text_a[0])
            print(input_ent)
            print(example.text_a[0])
            exit()
        if input_ids.count(1601) != 1 or input_ids.count(1089) != 1:
            print(tokens_a)
            print('---')
            print(input_ids)
            print('---')
            print('∞:', input_ids.count(1601), ';', 'º:', input_ids.count(1089)
                )
            print('1601:', input_ids.count(1601), ';', 'º:', input_ids.
                count(1089))
            print('=======')
        features.append(InputFeatures(input_ids=input_ids, input_mask=
            input_mask, segment_ids=segment_ids, input_ent=input_ent,
            ent_mask=ent_mask, labels=labels))
    return features
