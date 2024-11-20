@property
def __dict__(self):
    super_fields = {key: getattr(self, key) for key in DiffusionRegion.
        __dataclass_fields__.keys()}
    return {**super_fields, 'reference_image': self.reference_image.cpu().
        tolist(), 'strength': self.strength}
