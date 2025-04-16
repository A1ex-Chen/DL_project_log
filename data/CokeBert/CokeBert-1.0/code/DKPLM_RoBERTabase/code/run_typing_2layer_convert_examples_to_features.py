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
        if 'CLSID ' in ex_text_a:
            print('=======')
            print('CLSID in the setence:')
            print(ex_text_a)
            print('=======')
        if ' UCHIJ' in ex_text_a:
            print('=======')
            print('UCHIJ in the setence:')
            print(ex_text_a)
            print('=======')
        ex_text_a = ex_text_a[:h[1]] + 'CLSID ' + ex_text_a[h[1]:h[2]
            ] + ' UCHIJ' + ex_text_a[h[2]:]
        """
        begin, end = h[1:3]
        ### CLS and #TJ: begining of the entity and the end of end (entity range)
        #h[1] += 2
        #h[2] += 2
        h[1] += 6
        h[2] += 6
        ###
        """
        first_token = ex_text_a.split(' ')[0]
        if first_token == 'CLSID':
            tokens_a = tokenizer.tokenize(' ' + ex_text_a)
            """
            check_id_in_first_token+=1
            print(check_id_in_first_token)
            print(check_id_in_first_token/(ex_index+1))
            print("==================================")
            """
        else:
            tokens_a = tokenizer.tokenize(' ' + ex_text_a)
        CLSID_list = list()
        GCLSID_list = list()
        UCHIJ_list = list()
        GUCHIJ_list = list()
        entities_a = list()
        CLSID_list = list()
        GCLSID_list = list()
        UCHIJ_list = list()
        GUCHIJ_list = list()
        entities_a = list()
        typing_flag = False
        entities_a = ['UNK'] * len(tokens_a)
        for i_th, token in enumerate(tokens_a):
            if 'CLSID' == token:
                CLSID_list.append(token)
            if 'ĠCLSID' == token:
                GCLSID_list.append(token)
            if 'UCHIJ' == token:
                UCHIJ_list.append(token)
            if 'ĠUCHIJ' == token:
                GUCHIJ_list.append(token)
            if typing_flag == True:
                entities_a[i_th] = h[0]
            if token == 'CLSID' or token == 'ĠCLSID':
                typing_flag = True
            if token == 'UCHIJ' or token == 'ĠUCHIJ':
                typing_flag = False
                entities_a[i_th] = 'UNK'
        if len(CLSID_list) + len(GCLSID_list) != 1:
            print('=======')
            print('CLSID Wrong')
            print('=======')
            print(ex_text_a)
            print('---')
            print(tokens_a)
            print('---')
            print(entities_a)
            entities_a = ['UNK'] * len(tokens_a)
            CLSID_id_list = list()
            UCHIJ_id_list = list()
            for i_th, token in enumerate(tokens_a):
                if token == 'ĠCLSID':
                    CLSID_id_list.append(i_th)
                if token == 'ĠUCHIJ':
                    UCHIJ_id_list.append(i_th)
                    break
            CLSID_id = CLSID_id_list[-1] + 1
            UCHIJ_id = UCHIJ_id_list[-1]
            entities_a[CLSID_id:UCHIJ_id] = h[0] * (UCHIJ_id - CLSID_id)
            for CLSID_id in CLSID_id_list[:-1]:
                tokens_a[CLSID_id] = 'CLSID'
        if len(UCHIJ_list) + len(GUCHIJ_list) != 1:
            print('=======')
            print('UCHIJ Wrong')
            print('=======')
            print(ex_text_a)
            print('---')
            print(tokens_a)
            print('---')
            print(entities_a)
            entities_a = ['UNK'] * len(tokens_a)
            CLSID_id_list = list()
            UCHIJ_id_list = list()
            for i_th, token in enumerate(tokens_a):
                if token == 'ĠCLSID':
                    CLSID_id_list.append(i_th)
                if token == 'ĠUCHIJ':
                    UCHIJ_id_list.append(i_th)
                    break
            CLSID_id = CLSID_id_list[-1] + 1
            UCHIJ_id = UCHIJ_id_list[-1]
            entities_a[CLSID_id:UCHIJ_id] = h[0] * (UCHIJ_id - CLSID_id)
            for UCHIJ_id in UCHIJ_id_list[:-1]:
                tokens_a[UCHIJ_id] = 'UCHIJ'
        ent_pos = [x for x in example.text_b if x[-1] > threshold]
        for x in ent_pos:
            x[-1] = example.text_a[0][x[1]:x[2]]
        entities = ['UNK'] * len(tokens_a)
        for x in ent_pos:
            res = tokenizer.tokenize(' ' + x[-1])
            pos = 0
            mark = False
            while res[0] in tokens_a[pos:]:
                idx = tokens_a.index(res[0], pos)
                if check_pre(tokens_a[idx:], res):
                    entities[idx] = x[0]
                    mark = True
                    break
                else:
                    pos = idx + 1
            if mark:
                continue
            old_res = res
            res = tokenizer.tokenize(x[-1])
            pos = 0
            while res[0] in tokens_a[pos:]:
                idx = tokens_a.index(res[0], pos)
                if check_pre(tokens_a[idx:], res):
                    entities[idx] = x[0]
                    mark = True
                    break
                else:
                    pos = idx + 1
            if not mark:
                print(old_res)
                print(res)
                print(tokens_a)
                assert mark
        if h[1] == h[2]:
            print('h[1]==h[2]')
            exit()
            continue
        mark = False
        tokens_b = None
        for e in entities_a:
            if e != 'UNK':
                mark = True
                break
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
            entities_a = entities_a[:max_seq_length - 2]
            entities = entities[:max_seq_length - 2]
        tokens = ['<s>'] + tokens_a + ['</s>']
        real_ents = ['UNK'] + entities + ['UNK']
        ents = ['UNK'] + entities_a + ['UNK']
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
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
            entities_a = entities_a[:max_seq_length - 2]
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        ent_mask += padding
        input_ent += padding_
        input_ids_ = torch.LongTensor(input_ids)
        if len(input_ids_[input_ids_ == 50001]) != 1 or len(input_ids_[
            input_ids_ == 50210]) != 1:
            print("Less or more than 1 'UCHIJ' or 'CLSID' ")
            print(ex_text_a)
            print(tokens_a)
            print(entities)
            exit()
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(ent_mask) == max_seq_length
        assert len(input_ent) == max_seq_length
        labels = [0] * len(label_map)
        for l in example.label:
            l = label_map[l]
            labels[l] = 1
        if ex_index < 10:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('Entity: %s' % example.text_a[1])
            logger.info('tokens: %s' % ' '.join([str(x) for x in zip(tokens,
                ents)]))
            logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids])
                )
            logger.info('label: %s %s' % (example.label, labels))
            logger.info(real_ents)
        features.append(InputFeatures(input_ids=input_ids, input_mask=
            input_mask, segment_ids=segment_ids, input_ent=input_ent,
            ent_mask=ent_mask, labels=labels))
        """
        print(ex_text_a)
        print("---")
        print(tokens)
        print(len(tokens))
        print("---")
        print(ent_pos)
        print("---")
        print(ent_mask)
        #print(ent_mask.index(0))
        ent_mask_ = torch.LongTensor(ent_mask)
        print(ent_mask_.nonzero())
        print("---")
        print(input_ent)
        print("---")
        print(input_mask)
        print(input_mask.index(0))
        print("---")
        print(input_ids)
        print("---")
        print("tokens --> string")

        print("====================")
        print("====================")
        """
    return features
