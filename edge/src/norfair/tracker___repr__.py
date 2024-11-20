def __repr__(self):
    if self.last_distance is None:
        placeholder_text = (
            '\x1b[1mObject_{}\x1b[0m(age: {}, hit_counter: {}, last_distance: {}, init_id: {})'
            )
    else:
        placeholder_text = (
            '\x1b[1mObject_{}\x1b[0m(age: {}, hit_counter: {}, last_distance: {:.2f}, init_id: {})'
            )
    return placeholder_text.format(self.id, self.age, self.hit_counter,
        self.last_distance, self.initializing_id)
