def sums(self, format_string='{}'):
    return {format_string.format(name): meter.sum for name, meter in self.
        meters.items()}
