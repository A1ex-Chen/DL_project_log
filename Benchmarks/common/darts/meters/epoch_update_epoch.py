def update_epoch(self):
    self.loss.append(self.loss_meter.avg)
    for task, acc in self.acc_meter.meters.items():
        self.acc[task].append(acc.avg)
