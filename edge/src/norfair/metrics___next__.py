def __next__(self):
    if self.frame_number <= self.length:
        self.frame_number += 1
        return self.sorted_by_frame[self.frame_number - 2]
    raise StopIteration()
