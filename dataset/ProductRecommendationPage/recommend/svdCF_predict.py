def predict(self, uid, n):
    datamat = get_user_item_matrix(self.df)
    columns = datamat.columns
    datamat = np.asmatrix(datamat)
    N_arr = np.nonzero(datamat[uid, :] == 0.0)[1]
    if len(N_arr) == 0:
        print('you rated everything')
        return
    item_score = {}
    for i in N_arr:
        s = self.predict_oneuser(uid, columns[i])
        item_score[columns[i]] = s
    recs = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))[:
        n]
    return recs
