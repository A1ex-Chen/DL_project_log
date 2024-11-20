def __init__(self, tasks, name='train'):
    self.name = name
    self.loss_meter = AverageMeter(name)
    self.acc_meter = MultitaskAccuracyMeter(tasks)
    self.reset()
