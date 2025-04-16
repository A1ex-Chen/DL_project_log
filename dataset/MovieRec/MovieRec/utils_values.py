def values(self, format_string='{}'):
    return {format_string.format(name): meter.val for name, meter in self.
        meters.items()}
