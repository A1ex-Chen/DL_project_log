@property
def scheduler(self):
    return self._scheduler or self.trainer.scheduler
