def __eq__(self, other):
    if not isinstance(other, AssetSpec):
        return False
    return (self.name == other.name and self.version == other.version and 
        self.sub_part == other.sub_part)
