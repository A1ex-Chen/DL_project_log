@property
def reid_hit_counter_is_positive(self):
    return self.reid_hit_counter is None or self.reid_hit_counter >= 0
