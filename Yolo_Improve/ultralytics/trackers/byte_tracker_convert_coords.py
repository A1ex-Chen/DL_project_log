def convert_coords(self, tlwh):
    """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
    return self.tlwh_to_xyah(tlwh)
