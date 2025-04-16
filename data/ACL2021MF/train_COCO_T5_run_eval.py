def run_eval(_C, model, eval_data_iter, tokenzier, copy_vocab, device,
    output_path=None, test=False, full_eval=False, decode_constraint=None,
    novel_constraint_path=None):
    model.eval()
    predictions = []
    gen, gts, img_ids = [], [], []
    mentioned_cls = []
    novel_mentioned_cls = []
    used_cls = {}
    macro_mention = [0, 0]
    novel_macro_mention = [0, 0]
    novel_constraints = []
    if novel_constraint_path is not None:
        with open(novel_constraint_path) as out:
            for l in out:
                novel_constraints.append(int(l.strip()))
    with torch.no_grad():
        for batch in tqdm(eval_data_iter):
            for n in batch:
                if n in ['gt', 'image_ids']:
                    continue
                batch[n] = batch[n].to(device)
            encoder_cls = batch['encoder_cls'].detach().cpu().numpy()
            mention_flag = batch['mention_flag'].detach().cpu().numpy()
            cls_used = []
            for b_idx in range(encoder_cls.shape[0]):
                e_cls = encoder_cls[b_idx].tolist()
                mf = mention_flag[b_idx, 0].tolist()
                visited_cls = set()
                for cls_, m in zip(e_cls, mf):
                    if m == 1:
                        visited_cls.add(cls_)
                cls_used.append(list(visited_cls))
            if decode_constraint is not None:
                constraint_dict = {}
                for i, image_id in enumerate(batch['image_ids']):
                    constraint_dict[image_id] = []
                    for cls_index in cls_used[i]:
                        c = []
                        for _, fg_idx in copy_vocab.d_to_w_group[cls_index]:
                            c.append(copy_vocab.token_fg_w[fg_idx])
                        constraint_dict[image_id].append(c)
                state_transform_list = []
                state_num_list = []
                for image_id in batch['image_ids']:
                    state_matrix, state_num = (decode_constraint.
                        get_state_matrix(_C.vocab_size, constraint_dict[
                        image_id], image_id))
                    state_transform_list.append(state_matrix)
                    state_num_list.append(state_num)
                max_size = max(state_num_list)
                state_transform_list = [s[:, :max_size, :max_size] for s in
                    state_transform_list]
                state_transition = torch.from_numpy(np.concatenate(
                    state_transform_list, axis=0)).bool().to(device)
            else:
                state_transition = None
            outputs = model.search(input_ids=batch['encoder_input_ids'],
                attention_mask=batch['encoder_mask'], encoder_img_mask=
                batch['encoder_img_mask'], encoder_obj_feature=batch[
                'encoder_obj_feature'], encoder_obj_box=batch[
                'encoder_obj_box'], encoder_relative_pos_index=batch[
                'encoder_rel_position'], decoder_mention_flag=batch[
                'mention_flag'], decoder_cls_on_input=batch['encoder_cls'],
                state_transition=state_transition, num_beams=5,
                length_penalty=0.6, max_length=_C.max_generation_len,
                min_length=2, no_repeat_ngram_size=3, early_stopping=True)
            if decode_constraint is not None:
                outputs = decode_constraint.select_state_func(outputs,
                    batch['image_ids'])
            out = [tokenizer.decode(g, skip_special_tokens=True,
                clean_up_tokenization_spaces=False) for g in outputs]
            gen += out
            if not test:
                gts += batch['gt']
            img_ids += batch['image_ids']
            for b_idx in range(encoder_cls.shape[0]):
                cls_count = 0
                total_count = 0
                novel_total_count = 0
                novel_cls_count = 0
                single_img_used_cls = []
                for cls_ in cls_used[b_idx]:
                    total_count += 1
                    if cls_ in novel_constraints:
                        novel_total_count += 1
                    for w, _ in copy_vocab.d_to_w_group[cls_]:
                        if w in out[b_idx]:
                            cls_count += 1
                            if cls_ in novel_constraints:
                                novel_cls_count += 1
                            single_img_used_cls.append(cls_)
                            break
                used_cls[batch['image_ids'][b_idx]] = single_img_used_cls
                macro_mention[0] += cls_count
                macro_mention[1] += total_count
                if total_count > 0:
                    mentioned_cls.append(100 * cls_count / total_count)
                novel_macro_mention[0] += novel_cls_count
                novel_macro_mention[1] += novel_total_count
                if novel_total_count > 0:
                    novel_mentioned_cls.append(100 * cls_count / total_count)
    for c in gen[:20]:
        print(c)
    predictions = []
    for img_id, p in zip(img_ids, gen):
        predictions.append({'image_id': img_id, 'caption': p})
    if output_path is not None:
        with open(output_path, 'w') as out:
            out.write(json.dumps(predictions) + '\n')
    with open('used_cls.txt', 'w') as out:
        for c in used_cls:
            list_ = [c] + used_cls[c]
            list_ = [str(s) for s in list_]
            out.write(','.join(list_) + '\n')
    if len(mentioned_cls) > 0 and macro_mention[1] > 0:
        print('Averaged Mentione Ratio %.2f' % (sum(mentioned_cls) / len(
            mentioned_cls)))
        print('Macro Mentione Ratio %.2f' % (100 * macro_mention[0] /
            macro_mention[1]))
    if len(novel_constraints) > 0:
        print('Averaged Novel Mentione Ratio %.2f' % (sum(
            novel_mentioned_cls) / len(novel_mentioned_cls)))
        print('Macro Novel Mentione Ratio %.2f' % (100 *
            novel_macro_mention[0] / novel_macro_mention[1]))
    if not test:
        if not _C.external_eval:
            gts = evaluation.PTBTokenizer.tokenize(gts)
            gen = evaluation.PTBTokenizer.tokenize(gen)
            diversity_sen = [v[0].split() for _, v in gen.items()]
            print('Diversity-1 %.2f' % distinct_n(diversity_sen, 1))
            print('Diversity-2 %.2f' % distinct_n(diversity_sen, 2))
            val_bleu, _ = evaluation.Bleu(n=4).compute_score(gts, gen)
            method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
            metric_dict = {}
            for metric, score in zip(method, val_bleu):
                metric_dict[metric] = {'entire': score * 100}
                print('%s %.2f' % (metric, score * 100))
            val_cider, _ = evaluation.Cider().compute_score(gts, gen)
            print('CIDEr %.2f' % (val_cider * 100))
            metric_dict['CIDEr'] = {'entire': val_cider}
            if full_eval:
                val_spice, _ = evaluation.Spice().compute_score(gts, gen)
                print('SPICE %.2f' % (val_spice * 100))
                val_meteor, _ = evaluation.Meteor().compute_score(gts, gen)
                print('METEOR %.2f' % (val_meteor * 100))
                val_rouge, _ = evaluation.Rouge().compute_score(gts, gen)
                print('ROUGE_L %.2f' % (val_rouge * 100))
        else:
            evaluator = NocapsEvaluator(phase='val' if val else 'test')
            metric_dict = evaluator.evaluate(predictions)
            for metric_name in metric_dict:
                for domain in metric_dict[metric_name]:
                    print(f'{metric_name} {domain}:', metric_dict[
                        metric_name][domain])
                print('')
    return metric_dict
