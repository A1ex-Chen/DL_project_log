def zero_grad(self, models, optimizer, how_to_zero):
    if how_to_zero == 'none':
        for model in models:
            for param in model.parameters():
                param.grad = None
    elif how_to_zero == 'model':
        for model in models:
            model.zero_grad()
    elif how_to_zero == 'optimizer':
        optimizer.zero_grad()
