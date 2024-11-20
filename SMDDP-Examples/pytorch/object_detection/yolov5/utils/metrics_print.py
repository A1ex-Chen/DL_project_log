def print(self):
    for i in range(self.nc + 1):
        print(' '.join(map(str, self.matrix[i])))
