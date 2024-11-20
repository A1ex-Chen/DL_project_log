def calLabel(article, abstract):
    hyps_list = article
    refer = ' '.join(abstract)
    scores = []
    rouge_scorer = Rouge()
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer, rouge_scorer)
        scores.append(mean_score)
    selected = [int(np.argmax(scores))]
    selected_sent_cnt = 1
    best_rouge = np.max(scores)
    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = '\n'.join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer, rouge_scorer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break
    return selected
