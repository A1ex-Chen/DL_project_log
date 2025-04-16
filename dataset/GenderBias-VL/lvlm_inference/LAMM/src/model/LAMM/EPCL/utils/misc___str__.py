def __str__(self):
    return self.fmt.format(median=self.median, avg=self.avg, global_avg=
        self.global_avg, max=self.max, value=self.value)
