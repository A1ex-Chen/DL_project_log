def svd_predict(data_df, user, n, simfunc=cosSim):
    datamat = np.asmatrix(data_df.values)
    N_arr = np.nonzero(datamat[user, :] == 0)[1]
    if len(N_arr) == 0:
        print('you rated everything')
        return
    item_score = predict_recommend(datamat, user, N_arr, simfunc)
    recs = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))[:
        n]
    return recs
