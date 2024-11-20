def convert_coords(self, tlwh):
    """Converts Top-Left-Width-Height bounding box coordinates to X-Y-Width-Height format."""
    return self.tlwh_to_xywh(tlwh)
