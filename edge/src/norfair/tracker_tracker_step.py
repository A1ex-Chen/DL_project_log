def tracker_step(self):
    if self.reid_hit_counter is None:
        if self.hit_counter <= 0:
            self.reid_hit_counter = self.reid_hit_counter_max
    else:
        self.reid_hit_counter -= 1
    self.hit_counter -= 1
    self.point_hit_counter -= 1
    self.age += 1
    self.filter.predict()
