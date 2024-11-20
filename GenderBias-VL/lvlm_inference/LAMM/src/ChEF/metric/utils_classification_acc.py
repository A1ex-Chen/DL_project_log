def classification_acc(gt_text, pred_text, syn_judge=True):
    words = pred_text.split(' ')
    if syn_judge:
        syn_set = wn.synsets(gt_text)
        for word in words:
            if word in voc_syndict:
                words.append(voc_syndict[word])
        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(word, 'n') for word in words]
        syn_list = []
        for syn in syn_set:
            syn_list += syn.lemma_names()
        for syn in syn_list:
            if syn in words:
                return True
    elif gt_text in words:
        return True
    return False
