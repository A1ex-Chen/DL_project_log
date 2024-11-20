@property
def num_anchors_per_location(self):
    num = 0
    for a_generator in self._anchor_generators:
        num += a_generator.num_anchors_per_localization
    return num
