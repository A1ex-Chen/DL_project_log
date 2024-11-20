def items_by_popularity(self):
    popularity = Counter()
    for user in range(self.user_count):
        popularity.update(self.train[user])
        popularity.update(self.val[user])
        popularity.update(self.test[user])
    popular_items = sorted(popularity, key=popularity.get, reverse=True)
    return popular_items
