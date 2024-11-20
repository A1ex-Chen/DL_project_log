def filter(self, record):
    record.rank = self.rank
    if self.log_all_ranks:
        return True
    else:
        return self.rank == 0
