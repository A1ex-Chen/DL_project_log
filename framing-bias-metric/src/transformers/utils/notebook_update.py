def update(self, value: int, force_update: bool=False, comment: str=None):
    """
        The main method to update the progress bar to :obj:`value`.

        Args:

            value (:obj:`int`):
                The value to use. Must be between 0 and :obj:`total`.
            force_update (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force and update of the internal state and display (by default, the bar will wait for
                :obj:`value` to reach the value it predicted corresponds to a time of more than the :obj:`update_every`
                attribute since the last update to avoid adding boilerplate).
            comment (:obj:`str`, `optional`):
                A comment to add on the left of the progress bar.
        """
    self.value = value
    if comment is not None:
        self.comment = comment
    if self.last_value is None:
        self.start_time = self.last_time = time.time()
        self.start_value = self.last_value = value
        self.elapsed_time = self.predicted_remaining = None
        self.first_calls = self.warmup
        self.wait_for = 1
        self.update_bar(value)
    elif value <= self.last_value and not force_update:
        return
    elif force_update or self.first_calls > 0 or value >= min(self.
        last_value + self.wait_for, self.total):
        if self.first_calls > 0:
            self.first_calls -= 1
        current_time = time.time()
        self.elapsed_time = current_time - self.start_time
        self.average_time_per_item = self.elapsed_time / (value - self.
            start_value)
        if value >= self.total:
            value = self.total
            self.predicted_remaining = None
            if not self.leave:
                self.close()
        else:
            self.predicted_remaining = self.average_time_per_item * (self.
                total - value)
        self.update_bar(value)
        self.last_value = value
        self.last_time = current_time
        self.wait_for = max(int(self.update_every / self.
            average_time_per_item), 1)
