def save_to_json(self):
    """Saves rescaled bounding boxes to 'bounding_boxes.json' based on image-to-canvas size ratio."""
    canvas_width, canvas_height = self.canvas.winfo_width(
        ), self.canvas.winfo_height()
    width_scaling_factor = self.img_width / canvas_width
    height_scaling_factor = self.img_height / canvas_height
    bounding_boxes_data = []
    for box in self.bounding_boxes:
        rescaled_box = []
        for x, y in box:
            rescaled_x = int(x * width_scaling_factor)
            rescaled_y = int(y * height_scaling_factor)
            rescaled_box.append((rescaled_x, rescaled_y))
        bounding_boxes_data.append({'points': rescaled_box})
    with open('bounding_boxes.json', 'w') as json_file:
        json.dump(bounding_boxes_data, json_file, indent=4)
    messagebox.showinfo('Success',
        'Bounding boxes saved to bounding_boxes.json')
