def averages(self, format_string='{}'):
    return {format_string.format(name): meter.avg for name, meter in self.
        meters.items()}
