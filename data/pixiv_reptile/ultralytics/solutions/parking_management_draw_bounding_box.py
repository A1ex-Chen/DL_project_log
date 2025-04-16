def draw_bounding_box(self, box):
    """
        Draw bounding box on canvas.

        Args:
            box (list): Bounding box data
        """
    for i in range(4):
        x1, y1 = box[i]
        x2, y2 = box[(i + 1) % 4]
        self.canvas.create_line(x1, y1, x2, y2, fill='blue', width=2)
