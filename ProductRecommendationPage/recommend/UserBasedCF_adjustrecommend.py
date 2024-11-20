def adjustrecommend(users_score_item, user_name):
    bookid_list = []
    rec = recommender(users_score_item)
    k, nearuser = rec.recommend(bytes(user_name))
    for i in range(len(k)):
        bookid_list.append(k[i][0])
    return bookid_list, nearuser[:7]
