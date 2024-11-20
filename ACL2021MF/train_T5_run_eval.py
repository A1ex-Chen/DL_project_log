def run_eval(_C, model, eval_data_iter, copy_vocab, tokenizer, device,
    decoder_start_token_id, only_test=False, decode_constraint=None,
    output_path=None, seen_constraint_path=None):
    model.eval()
    gts, pred, gt_concepts = [], [], []
    cls_recall = [0, 0]
    novel_cls_recall = [0, 0]
    seen_cls_recall = [0, 0]
    seen_constraint_list = []
    if seen_constraint_path is not None:
        with open(seen_constraint_path) as out:
            for l in out:
                l = l.strip()
                seen_constraint_list.append(l)
    with torch.no_grad():
        for batch in tqdm(eval_data_iter):
            for n in batch:
                if n not in ['gt', 'gt_concepts']:
                    batch[n] = batch[n].to(device)
            cls_used = []
            for i in range(batch['concept_cls'].size(0)):
                gt_cls = []
                for j in range(batch['concept_cls'].size(1)):
                    ix = batch['concept_cls'][i][j].item()
                    if ix > 0:
                        gt_cls.append(ix)
                cls_used.append(set(gt_cls))
            if decode_constraint is not None:
                constraint_dict = {}
                for i in range(batch['concept_cls'].size(0)):
                    constraint_dict[i] = []
                    for cls_index in cls_used[i]:
                        c = []
                        for _, fg_idx in copy_vocab.d_to_w_group[cls_index]:
                            c.append(copy_vocab.token_fg_w[fg_idx])
                        constraint_dict[i].append(c)
                state_transform_list = []
                state_num_list = []
                for i in range(batch['concept_cls'].size(0)):
                    state_matrix, state_num = (decode_constraint.
                        get_state_matrix(_C.vocab_size, constraint_dict[i], i))
                    state_transform_list.append(state_matrix)
                    state_num_list.append(state_num)
                max_size = max(state_num_list)
                state_transform_list = [s[:, :max_size, :max_size] for s in
                    state_transform_list]
                state_transition_np = np.concatenate(state_transform_list,
                    axis=0)
                state_transition = torch.from_numpy(state_transition_np).bool(
                    ).to(device)
            else:
                state_transition = None
            outputs = model.search(input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'], decoder_copy_pos=
                batch['copy_pos'], decoder_concept_cls=batch['concept_cls'],
                decoder_copy_mention_flag=batch['copy_mention_flag'],
                decoder_mention_flag=batch['decoder_mention_flag'],
                decoder_cls_on_input=batch['cls_on_input'],
                state_transition=state_transition, num_beams=5,
                length_penalty=1.0, max_length=25, min_length=2,
                no_repeat_ngram_size=3, early_stopping=True,
                decoder_start_token_id=decoder_start_token_id)
            if decode_constraint is not None:
                outputs = decode_constraint.select_state_func(outputs, [i for
                    i in range(batch['concept_cls'].size(0))])
            dec = [tokenizer.decode(g, skip_special_tokens=True,
                clean_up_tokenization_spaces=False) for g in outputs]
            for d, gt in zip(dec, batch['gt']):
                gts.append(gt)
                pred.append(d)
            gt_concepts += batch['gt_concepts']
            N, D = outputs.size()
            for i in range(N):
                gt_cls = cls_used[i]
                mention_cls = []
                if _C.use_pointer:
                    for j in range(D):
                        ix = outputs[i][j].item()
                        if ix >= _C.vocab_size:
                            ix = ix - _C.vocab_size
                            _cls = copy_vocab.i_to_cls[ix]
                            mention_cls.append(copy_vocab.id_to_category[_cls])
                else:
                    w_list = dec[i].split()
                    if w_list[-1].endswith('.'):
                        w_list[-1] = w_list[-1][:-1]
                    w_list = [(w[:-2] if w.endswith("'s") else w) for w in
                        w_list]
                    w_list = [(w[:-1] if w.endswith(',') else w) for w in
                        w_list]
                    for gt_c in gt_cls:
                        for w, _ in copy_vocab.d_to_w_group[gt_c]:
                            if w in w_list:
                                mention_cls.append(gt_c)
                                break
                mention_cls = set(mention_cls)
                novel_gt = set([c for c in gt_cls if copy_vocab.
                    id_to_category[c] not in seen_constraint_list])
                seen_gt = set([c for c in gt_cls if copy_vocab.
                    id_to_category[c] in seen_constraint_list])
                novel_mention = set([c for c in mention_cls if copy_vocab.
                    id_to_category[c] not in seen_constraint_list])
                seen_mention = set([c for c in mention_cls if copy_vocab.
                    id_to_category[c] in seen_constraint_list])
                cls_recall[1] += len(gt_cls)
                cls_recall[0] += len(gt_cls & mention_cls)
                novel_cls_recall[1] += len(novel_gt)
                seen_cls_recall[1] += len(seen_gt)
                novel_cls_recall[0] += len(novel_gt & novel_mention)
                seen_cls_recall[0] += len(seen_gt & seen_mention)
    for p in pred[:20]:
        print(p)
    if output_path is not None:
        output_list = []
        for _id, out in enumerate(pred):
            output_list.append({'image_id': _id, 'caption': out})
        with open(output_path, 'w') as out:
            out.write(json.dumps(output_list))
    gts = tokenize(gts)
    gen = tokenize(pred)
    coverage_score, overall_coverage = get_coverage_score(gt_concepts, pred)
    print('Coverage %.2f' % coverage_score)
    print('Macro Coverage %.2f' % overall_coverage)
    print('Token-Level Coverage %.2f' % (100 * cls_recall[0] / cls_recall[1]))
    if len(seen_constraint_list) > 0:
        print('Novel Token-Level Coverage %.2f' % (100 * novel_cls_recall[0
            ] / novel_cls_recall[1]))
        print('Seen Token-Level Coverage %.2f' % (100 * seen_cls_recall[0] /
            seen_cls_recall[1]))
    diversity_sen = [v[0].split() for _, v in gen.items()]
    print('Diversity-1 %.2f' % distinct_n(diversity_sen, 1))
    print('Diversity-2 %.2f' % distinct_n(diversity_sen, 2))
    val_bleu, _ = evaluation.Bleu(n=4).compute_score(gts, gen)
    method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    metric_dict = {}
    for metric, score in zip(method, val_bleu):
        metric_dict['metric'] = {'entire': score * 100}
        print('%s %.2f' % (metric, score * 100))
    val_meteor, _ = evaluation.Meteor().compute_score(gts, gen)
    print('METEOR %.2f' % (val_meteor * 100))
    val_rouge, _ = evaluation.Rouge().compute_score(gts, gen)
    print('ROUGE_L %.2f' % (val_rouge * 100))
    val_cider, _ = evaluation.Cider().compute_score(gts, gen)
    print('CIDEr %.2f' % (val_cider * 100))
    val_spice, _ = evaluation.Spice().compute_score(gts, gen)
    print('SPICE %.2f' % (val_spice * 100))
    metric_dict.update({'CIDEr': {'entire': val_cider}, 'ROUGE_L': {
        'entire': val_rouge}, 'METEOR': {'entire': val_meteor}, 'SPICE': {
        'entire': val_spice}})
    return metric_dict
