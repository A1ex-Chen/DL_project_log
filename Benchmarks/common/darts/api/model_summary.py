def summary(self, hashsummary=False):
    print(self)
    print('-' * 80)
    n_params = self.num_params()
    print(f'Number of model parameters: {n_params}')
    print('-' * 80)
    if hashsummary:
        print('Hash Summary:')
        for idx, hashvalue in enumerate(self.hashsummary()):
            print(f'{idx}: {hashvalue}')
