def train(self, train_data, k=25, num_iterations=4):
    self.initialize_factors(train_data, k)
    for iteration in tqdm(range(num_iterations)):
        self.error = self.loss()
        logger.debug('iteration {} loss {}'.format(iteration, self.error))
        for usr, pos, neg in self.draw(self.ratings.shape[0]):
            self.step(usr, pos, neg)
        self.save(iteration, iteration == num_iterations - 1)
