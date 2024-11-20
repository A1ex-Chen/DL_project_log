def train_or_evaluate(phrase1_tensor, phrase2_tensor, label_tensor,
    best_threshold=None):
    y_score = list(cosine_similarity(phrase1_tensor, phrase2_tensor).diagonal()
        )
    y_true = list(label_tensor)
    avg_cosine_similarity = np.mean(y_score)
    std_cosine_similarity = np.std(np.array(y_score))
    best_acc = 0
    if not best_threshold:
        for i in range(len(y_score)):
            th = y_score[i]
            y_test = y_score >= th
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_threshold = th
    else:
        y_test = y_score >= best_threshold
        best_acc = np.mean((y_test == y_true).astype(int))
    y_score_pos, y_true_pos = [], []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            y_score_pos.append(y_score[i])
            y_true_pos.append(y_true[i])
    y_test_pos = y_score_pos >= best_threshold
    best_acc_pos = np.mean((y_test_pos == y_true_pos).astype(int))
    return (best_threshold, best_acc, best_acc_pos, avg_cosine_similarity,
        std_cosine_similarity)
