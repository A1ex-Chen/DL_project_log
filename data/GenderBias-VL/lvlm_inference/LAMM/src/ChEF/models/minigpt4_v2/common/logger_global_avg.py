def global_avg(self):
    loss_str = []
    for name, meter in self.meters.items():
        loss_str.append('{}: {:.4f}'.format(name, meter.global_avg))
    return self.delimiter.join(loss_str)
