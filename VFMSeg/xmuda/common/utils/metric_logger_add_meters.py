def add_meters(self, meters):
    if not isinstance(meters, (list, tuple)):
        meters = [meters]
    for meter in meters:
        self.add_meter(meter.name, meter)
