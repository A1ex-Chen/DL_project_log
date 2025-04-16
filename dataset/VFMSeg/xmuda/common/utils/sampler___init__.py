def __init__(self, batch_sampler, num_iterations, start_iter=0):
    self.batch_sampler = batch_sampler
    self.num_iterations = num_iterations
    self.start_iter = start_iter
