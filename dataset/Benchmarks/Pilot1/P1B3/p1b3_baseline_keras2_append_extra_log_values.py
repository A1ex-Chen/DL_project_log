def append_extra_log_values(self, tuples):
    for k, v in tuples:
        self.extra_log_values.append((k, v))
