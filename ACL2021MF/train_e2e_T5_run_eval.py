def run_eval(_C, model, eval_data_iter, tokenizer, copy_vocab, device,
    decode_constraint=None, constraint_vocab=None, output_path=None):
    model.eval()
    if decode_constraint is not None:
        assert constraint_vocab is not None
        constraint_vocab_dict = {}
        with open(constraint_vocab) as out:
            for line in out:
                line = line.strip()
                items = line.split('@')
                constraint_vocab_dict[items[0]] = items[1:]
    gt_cap, pred = [], []
    obj_coverage = [0, 0]
    with torch.no_grad():
        for batch in tqdm(eval_data_iter):
            for n in batch:
                if n not in ['gt', 'gt_mr', 'ins_id']:
                    batch[n] = batch[n].to(device)
            if decode_constraint is not None:
                constraint_dict = {}
                for id_, gt_mr in enumerate(batch['gt_mr']):
                    constraint_dict[id_] = []
                    for mr, _ in gt_mr:
                        if mr in constraint_vocab_dict:
                            c = []
                            for fg_w in constraint_vocab_dict[mr]:
                                fg_index = copy_vocab.w_to_i[fg_w]
                                c.append(copy_vocab.token_fg_w[fg_index])
                            constraint_dict[id_].append(c)
                state_transform_list = []
                state_num_list = []
                for image_id in range(len(batch['gt_mr'])):
                    state_matrix, state_num = (decode_constraint.
                        get_state_matrix(_C.vocab_size, constraint_dict[
                        image_id], image_id))
                    state_transform_list.append(state_matrix)
                    state_num_list.append(state_num)
                max_size = max(state_num_list)
                state_transform_list = [s[:, :max_size, :max_size] for s in
                    state_transform_list]
                state_transition = np.concatenate(state_transform_list, axis=0)
                state_transition = torch.from_numpy(state_transition).bool(
                    ).to(device)
            else:
                state_transition = None
            outputs = model.search(input_ids=batch['encoder_input_ids'],
                attention_mask=batch['encoder_mask'], decoder_mention_flag=
                batch['mention_flag'], decoder_cls_on_input=batch[
                'encoder_cls'], state_transition=state_transition,
                num_beams=5, length_penalty=1.0, max_length=_C.
                max_generation_len, min_length=2, no_repeat_ngram_size=3,
                early_stopping=True)
            if decode_constraint is not None:
                outputs = decode_constraint.select_state_func(outputs, [i for
                    i in range(len(batch['gt_mr']))])
            dec = [tokenizer.decode(g, skip_special_tokens=True,
                clean_up_tokenization_spaces=False) for g in outputs]
            for ins_id, d, gt, gt_mr in zip(batch['ins_id'], dec, batch[
                'gt'], batch['gt_mr']):
                gt_cap.append(gt)
                pred.append((ins_id, d))
                gt_count = 0
                lower_d = d.lower()
                for fullname, g_class_name in gt_mr:
                    gt_count += 1
                    cls_id = copy_vocab.word_to_category_id[fullname]
                    has_found = False
                    for w, _ in copy_vocab.d_to_w_group[cls_id]:
                        if w.lower() in lower_d:
                            obj_coverage[0] += 1
                            has_found = True
                            break
                obj_coverage[1] += gt_count
    for p in pred[:20]:
        print(p)
    if output_path is not None:
        output_list = []
        for _id, out in pred:
            output_list.append({'image_id': _id, 'caption': out})
        with open(output_path, 'w') as out:
            out.write(json.dumps(output_list))
    pred = [p[1] for p in pred]
    gts = evaluation.PTBTokenizer.tokenize(gt_cap)
    gen = evaluation.PTBTokenizer.tokenize(pred)
    print('Object Coverage %.2f' % (100 * obj_coverage[0] / obj_coverage[1]))
    diversity_sen = [v[0].split() for _, v in gen.items()]
    print('Diversity-1 %.2f' % distinct_n(diversity_sen, 1))
    print('Diversity-2 %.2f' % distinct_n(diversity_sen, 2))
    bleu = BLEUScore()
    nist = NISTScore()
    for sents_ref, sent_sys in zip(gt_cap, pred):
        bleu.append(sent_sys, sents_ref)
        nist.append(sent_sys, sents_ref)
    print('NIST %.2f' % nist.score())
    print('BLEU %.2f' % (bleu.score() * 100))
    val_meteor, _ = evaluation.Meteor().compute_score(gts, gen)
    print('METEOR %.2f' % (val_meteor * 100))
    val_cider, individual_cider = evaluation.Cider().compute_score(gts, gen)
    print('CIDEr %.2f' % val_cider)
    val_rouge, _ = evaluation.Rouge().compute_score(gts, gen)
    print('ROUGE_L %.2f' % (val_rouge * 100))
    metric_dict = {'CIDEr': {'entire': val_cider}}
    metric_dict.update({'METEOR': {'entire': val_meteor}})
    return metric_dict
