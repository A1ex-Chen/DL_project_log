def mouse_event_for_distance(self, event, x, y, flags, param):
    """
        Handles mouse events to select regions in a real-time video stream.

        Args:
            event (int): Type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): X-coordinate of the mouse pointer.
            y (int): Y-coordinate of the mouse pointer.
            flags (int): Flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY, cv2.EVENT_FLAG_SHIFTKEY, etc.).
            param (dict): Additional parameters passed to the function.
        """
    if event == cv2.EVENT_LBUTTONDOWN:
        self.left_mouse_count += 1
        if self.left_mouse_count <= 2:
            for box, track_id in zip(self.boxes, self.trk_ids):
                if box[0] < x < box[2] and box[1] < y < box[3
                    ] and track_id not in self.selected_boxes:
                    self.selected_boxes[track_id] = box
    elif event == cv2.EVENT_RBUTTONDOWN:
        self.selected_boxes = {}
        self.left_mouse_count = 0
