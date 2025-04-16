def update_bar(self, value, comment=None):
    spaced_value = ' ' * (len(str(self.total)) - len(str(value))) + str(value)
    if self.elapsed_time is None:
        self.label = f'[{spaced_value}/{self.total} : < :'
    elif self.predicted_remaining is None:
        self.label = (
            f'[{spaced_value}/{self.total} {format_time(self.elapsed_time)}')
    else:
        self.label = (
            f'[{spaced_value}/{self.total} {format_time(self.elapsed_time)} < {format_time(self.predicted_remaining)}'
            )
        self.label += f', {1 / self.average_time_per_item:.2f} it/s'
    self.label += ']' if self.comment is None or len(self.comment
        ) == 0 else f', {self.comment}]'
    self.display()
