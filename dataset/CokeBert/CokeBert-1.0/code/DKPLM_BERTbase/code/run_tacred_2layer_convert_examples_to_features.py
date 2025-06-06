def convert_examples_to_features(examples, label_list, max_seq_length,
    tokenizer, threshold):
    """Loads a data file into a list of `InputBatch`s."""
    label_list = sorted(label_list)
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
        h, t = example.text_a[1]
        h_name = ex_text_a[h[1]:h[2]]
        t_name = ex_text_a[t[1]:t[2]]
        if ('∞' in ex_text_a or 'π' in ex_text_a or 'º' in ex_text_a or '∂' in
            ex_text_a):
            print(ex_text_a)
            print('Line 166')
            exit()
        """
        input_ids = tokenizer.convert_tokens_to_ids("∞")
        print(input_ids)
        print("======")
        input_ids = tokenizer.convert_tokens_to_ids("π")
        print(input_ids)
        print("======")
        input_ids = tokenizer.convert_tokens_to_ids("º")
        print(input_ids)
        print("======")
        input_ids = tokenizer.convert_tokens_to_ids("∂")
        print(input_ids)
        print("======")
        exit()
        """
        if h[1] < t[1]:
            ex_text_a = ex_text_a[:h[1]] + '∞ ' + h_name + ' π' + ex_text_a[h
                [2]:t[1]] + 'º ' + t_name + ' ∂' + ex_text_a[t[2]:]
        else:
            ex_text_a = ex_text_a[:t[1]] + 'º ' + t_name + ' ∂' + ex_text_a[t
                [2]:h[1]] + '∞ ' + h_name + ' π' + ex_text_a[h[2]:]
        ent_pos = [x for x in example.text_b if x[-1] > threshold]
        for x in ent_pos:
            cnt = 0
            if x[1] > h[2]:
                cnt += 2
            if x[1] >= h[1]:
                cnt += 2
            if x[1] >= t[1]:
                cnt += 2
            if x[1] > t[2]:
                cnt += 2
            x[1] += cnt
            x[2] += cnt
        tokens_a, entities_a = tokenizer.tokenize(ex_text_a, ent_pos)
        """
        cnt = 0
        for x in entities_a:
            if x != "UNK":
                cnt += 1
        if cnt != len(ent_pos) and ent_pos[0][0] != 'Q46809':
            print(cnt, len(ent_pos))
            print(ex_text_a)
            print(ent_pos)
            for x in ent_pos:
                print(ex_text_a[x[1]:x[2]])
            exit(1)
        """
        tokens_b = None
        if False:
            tokens_b, entities_b = tokenizer.tokenize(example.text_b[0], [x for
                x in example.text_b[1] if x[-1] > threshold])
            _truncate_seq_pair(tokens_a, tokens_b, entities_a, entities_b, 
                max_seq_length - 3)
        elif len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
            entities_a = entities_a[:max_seq_length - 2]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        ents = ['UNK'] + entities_a + ['UNK']
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            ents += entities_b + ['UNK']
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ent = []
        ent_mask = []
        for ent in ents:
            if ent != 'UNK' and ent in entity2id:
                input_ent.append(entity2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1
        input_mask = [1] * len(input_ids)
        """
        print(tokens)
        print("===")
        print(input_ids)
        exit()
        """
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        input_ent += padding_
        ent_mask += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_ent) == max_seq_length
        assert len(ent_mask) == max_seq_length
        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('tokens: %s' % ' '.join([str(x) for x in tokens]))
            logger.info('ents: %s' % ' '.join([str(x) for x in ents]))
            logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids])
                )
            logger.info('input_mask: %s' % ' '.join([str(x) for x in
                input_mask]))
            logger.info('segment_ids: %s' % ' '.join([str(x) for x in
                segment_ids]))
            logger.info('label: %s (id = %d)' % (example.label, label_id))
        if (1170 not in input_ids or 1601 not in input_ids or 1089 not in
            input_ids or 1592 not in input_ids):
            print('======')
            print(tokens)
            print('-----')
            print(input_ids)
            print('======')
            continue
        features.append(InputFeatures(input_ids=input_ids, input_mask=
            input_mask, segment_ids=segment_ids, input_ent=input_ent,
            ent_mask=ent_mask, label_id=label_id))
    return features
