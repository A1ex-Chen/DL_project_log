def get_ids(self) ->Tuple[int, int]:
    self.count += 1
    _TrackedObjectFactory.global_count += 1
    return self.count, _TrackedObjectFactory.global_count
