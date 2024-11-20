def nonlinear_transform(self, x):
    if self.activation == 'linear':
        return x
    elif self.activation == 'sigmoid':
        return torch.sigmoid(x)
