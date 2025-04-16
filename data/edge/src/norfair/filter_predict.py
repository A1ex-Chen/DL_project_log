def predict(self):
    self.x[:self.dim_z] += self.x[self.dim_z:]
