def get_initializing_id(self) ->int:
    self.initializing_count += 1
    return self.initializing_count
