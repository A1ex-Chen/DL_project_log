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
        assert 'UKIP' not in ex_text_a
        assert 'CLSID' not in ex_text_a
        if h[1] < t[1]:
            ex_text_a = ex_text_a[:h[1]] + '# ' + h_name + ' UKIP' + ex_text_a[
                h[2]:t[1]] + '$ ' + t_name + ' CLSID' + ex_text_a[t[2]:]
        else:
            ex_text_a = ex_text_a[:t[1]
                ] + '$ ' + t_name + ' CLSID' + ex_text_a[t[2]:h[1]
                ] + '# ' + h_name + ' UKIP' + ex_text_a[h[2]:]
        if h[1] < t[1]:
            h[1] += 2
            h[2] += 2
            t[1] += 6
            t[2] += 6
        else:
            h[1] += 6
            h[2] += 6
            t[1] += 2
            t[2] += 2
        first_token = ex_text_a.split(' ')[0]
        if first_token == '#':
            tokens_a = tokenizer.tokenize(' ' + ex_text_a)
        else:
            tokens_a = tokenizer.tokenize(ex_text_a)
        entities_a = ['UNK'] * len(tokens_a)
        for i in range(len(tokens_a)):
            if tokens_a[i] == 'Ġ#' or tokens_a[i] == '#':
                entities_a[i + 1] = h[0]
                break
        for i in range(len(tokens_a)):
            if tokens_a[i] == 'Ġ$' or tokens_a[i] == '$':
                entities_a[i + 1] = t[0]
                break
        if len([x for x in entities_a if x != 'UNK']) != 2:
            print(ex_text_a)
            print('--')
            print(tokens_a)
            print('--')
            print(entities_a)
            print('--')
            print(len([x for x in entities_a if x[0] != 'UNK']))
            exit(1)
        tokens_b = None
        if example.text_b:
            tokens_b, entities_b = tokenizer.tokenize(example.text_b[0], [x for
                x in example.text_b[1] if x[-1] > threshold])
            _truncate_seq_pair(tokens_a, tokens_b, entities_a, entities_b, 
                max_seq_length - 3)
        elif len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
            entities_a = entities_a[:max_seq_length - 2]
        tokens = ['<s>'] + tokens_a + ['</s>']
        ents = ['UNK'] + entities_a + ['UNK']
        segment_ids = [0] * len(tokens)
        """
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            ents += entities_b + ["UNK"]
            segment_ids += [1] * (len(tokens_b) + 1)
        """
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
        for i, id in enumerate(input_ids):
            if 10431 == id:
                input_ids[i] = 849
            if 1629 == id:
                input_ids[i] = 68
        if 849 not in input_ids or 68 not in input_ids:
            print(input_ids)
            print('Still no 68 or 849')
            exit()
            continue
        input_ids_tensor = torch.LongTensor(input_ids)
        if len(input_ids_tensor[input_ids_tensor == 849]) > 1 or len(
            input_ids_tensor[input_ids_tensor == 68]) > 1:
            print('More than one 68 or 849')
            print('===')
            print(ex_text_a)
            print('--')
            print(tokens)
            print('--')
            print(input_ids)
            print('---')
            if len(input_ids_tensor[input_ids_tensor == 849]) > 1:
                id_G = input_ids.index(849)
                j = id_G
                locate = list()
                while j < len(input_ids):
                    if input_ids[j] == 35829:
                        loc = locate[-1]
                        z = 0
                        while z < len(input_ids):
                            if input_ids[z] == 849:
                                input_ids[z] = 0
                            z += 1
                        input_ids[loc] = 849
                        break
                    elif input_ids[j] == 849:
                        locate.append(j)
                    j += 1
            if len(input_ids_tensor[input_ids_tensor == 68]) > 1:
                id_G = input_ids.index(68)
                j = id_G
                locate = list()
                locate.append(j)
                while j < len(input_ids):
                    if input_ids[j] == 50001:
                        loc = locate[-1]
                        z = 0
                        while z < len(input_ids):
                            if input_ids[z] == 68:
                                input_ids[z] = 0
                            z += 1
                        input_ids[loc] = 68
                        break
                    elif input_ids[j] == 68:
                        locate.append(j)
                    j += 1
            print(input_ids)
            print('===')
            print('---')
        features.append(InputFeatures(input_ids=input_ids, input_mask=
            input_mask, segment_ids=segment_ids, input_ent=input_ent,
            ent_mask=ent_mask, label_id=label_id))
    return features
