def classification_acc(gt_text, pred_text):
    words = parse_entity(pred_text)
    syn_set = wn.synsets(gt_text)
    try:
        syn_list = syn_set[0].lemma_names()
    except:
        syn_list = [gt_text]
    for syn in syn_list:
        if syn in words:
            return True
    return False
