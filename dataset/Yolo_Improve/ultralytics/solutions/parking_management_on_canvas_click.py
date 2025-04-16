def on_canvas_click(self, event):
    """Handle mouse clicks on canvas to create points for bounding boxes."""
    self.current_box.append((event.x, event.y))
    x0, y0 = event.x - 3, event.y - 3
    x1, y1 = event.x + 3, event.y + 3
    self.canvas.create_oval(x0, y0, x1, y1, fill='red')
    if len(self.current_box) == 4:
        self.bounding_boxes.append(self.current_box)
        self.draw_bounding_box(self.current_box)
        self.current_box = []
