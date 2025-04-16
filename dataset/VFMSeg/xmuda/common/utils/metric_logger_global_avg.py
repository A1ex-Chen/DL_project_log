@property
def global_avg(self):
    return self.sum / self.count if self.count != 0 else float('nan')
