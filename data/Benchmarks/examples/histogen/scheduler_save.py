def save(self, filename):
    with open(filename, 'w') as f:
        for lr, loss in zip(self.lrs, self.losses):
            f.write('{},{}\n'.format(lr, loss))
