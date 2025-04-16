def draw(self, no):
    for _ in range(no):
        u = random.choice(self.user_ids)
        user_items = self.user_item[u]
        pos = random.choice(user_items)
        neg = pos
        while neg in user_items:
            neg = random.choice(self.item_ids)
        yield self.u_inx[u], self.i_inx[pos], self.i_inx[neg]
