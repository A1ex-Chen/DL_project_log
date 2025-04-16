def __format__(self, format):
    return '{self.val:{format}} ({self.avg:{format}})'.format(self=self,
        format=format)
