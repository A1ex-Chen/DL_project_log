def GetRecommendation(user):
    items = {}
    seen_items = set(train[user])
    for u, _ in sorted_user_sim[user][:K]:
        for item in train[u]:
            if item not in seen_items:
                if item not in items:
                    items[item] = 0
                items[item] += sim[user][u]
    recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
    return recs
