def build(self, ratings, params):
    if params:
        k = params['k']
        num_iterations = params['num_iterations']
    self.train(ratings, k, num_iterations)
