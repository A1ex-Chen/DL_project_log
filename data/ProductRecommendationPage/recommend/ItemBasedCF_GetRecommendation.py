def GetRecommendation(user):
    items = {}
    seen_items = set(train[user])
    for item in train[user]:
        for u, _ in sorted_item_sim[item][:K]:
            if u not in seen_items:
                if u not in items:
                    items[u] = 0
                items[u] += sim[item][u]
    recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
    return recs
