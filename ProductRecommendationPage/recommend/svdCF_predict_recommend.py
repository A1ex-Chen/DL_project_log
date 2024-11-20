def predict_recommend(datamat, user, N_arr, simfunc):
    item_score = {}
    for i in tqdm.tqdm(N_arr):
        score = svdForRecom(datamat, user, simfunc, i)
        if score <= 5.0 and score >= 0.0:
            item_score[int(i + 1)] = score
        else:
            item_score[int(i + 1)] = 0
    return item_score
