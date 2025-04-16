def __str__(self):
    loss_str = []
    for name, meter in self.meters.items():
        loss_str.append('{}: {}'.format(name, str(meter)))
    return self.delimiter.join(loss_str)
