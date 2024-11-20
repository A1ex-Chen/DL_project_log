def loss(self):
    br = self.bias_regularization
    ur = self.user_regularization
    pir = self.positive_item_regularization
    nir = self.negative_item_regularization
    ranking_loss = 0
    for u, i, j in self.loss_samples:
        x = self.predict(u, i) - self.predict(u, j)
        ranking_loss += 1.0 / (1.0 + exp(x))
    c = 0
    for u, i, j in self.loss_samples:
        c += ur * np.dot(self.user_factors[u], self.user_factors[u])
        c += pir * np.dot(self.item_factors[i], self.item_factors[i])
        c += nir * np.dot(self.item_factors[j], self.item_factors[j])
        c += br * self.item_bias[i] ** 2
        c += br * self.item_bias[j] ** 2
    return ranking_loss + 0.5 * c
