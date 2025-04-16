def predict(self, user, item):
    i_fac = self.item_factors[item]
    u_fac = self.user_factors[user]
    pq = i_fac.dot(u_fac)
    return pq + self.item_bias[item]
