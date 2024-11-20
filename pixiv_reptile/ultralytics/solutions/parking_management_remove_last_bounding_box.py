def remove_last_bounding_box(self):
    """Remove the last drawn bounding box from canvas."""
    if self.bounding_boxes:
        self.bounding_boxes.pop()
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.
            canvas_image)
        for box in self.bounding_boxes:
            self.draw_bounding_box(box)
        messagebox.showinfo('Success', 'Last bounding box removed.')
    else:
        messagebox.showwarning('Warning', 'No bounding boxes to remove.')
