def step(self, u, i, j):
    lr = self.learning_rate
    ur = self.user_regularization
    br = self.bias_regularization
    pir = self.positive_item_regularization
    nir = self.negative_item_regularization
    ib = self.item_bias[i]
    jb = self.item_bias[j]
    u_dot_i = np.dot(self.user_factors[u, :], self.item_factors[i, :] -
        self.item_factors[j, :])
    x = ib - jb + u_dot_i
    z = 1.0 / (1.0 + exp(x))
    ib_update = z - br * ib
    self.item_bias[i] += lr * ib_update
    jb_update = -z - br * jb
    self.item_bias[j] += lr * jb_update
    update_u = (self.item_factors[i, :] - self.item_factors[j, :]
        ) * z - ur * self.user_factors[u, :]
    self.user_factors[u, :] += lr * update_u
    update_i = self.user_factors[u, :] * z - pir * self.item_factors[i, :]
    self.item_factors[i, :] += lr * update_i
    update_j = -self.user_factors[u, :] * z - nir * self.item_factors[j, :]
    self.item_factors[j, :] += lr * update_j
