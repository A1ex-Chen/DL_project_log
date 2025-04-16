def __gt__(self, other):
    self_nums = [self.major, self.minor, self.patch]
    other_nums = [other.major, other.minor, other.patch]
    for self_ver, other_ver in zip(self_nums, other_nums):
        if self_ver > other_ver:
            return True
        elif self_ver < other_ver:
            return False
    return False
