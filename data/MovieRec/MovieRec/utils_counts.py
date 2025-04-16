def counts(self, format_string='{}'):
    return {format_string.format(name): meter.count for name, meter in self
        .meters.items()}
