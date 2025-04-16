def mouse_event_for_region(self, event, x, y, flags, params):
    """
        Handles mouse events for defining and moving the counting region in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any associated event flags (e.g., cv2.EVENT_FLAG_CTRLKEY,  cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters for the function.
        """
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(self.reg_pts):
            if isinstance(point, (tuple, list)) and len(point) >= 2 and (
                abs(x - point[0]) < 10 and abs(y - point[1]) < 10):
                self.selected_point = i
                self.is_drawing = True
                break
    elif event == cv2.EVENT_MOUSEMOVE:
        if self.is_drawing and self.selected_point is not None:
            self.reg_pts[self.selected_point] = x, y
            self.counting_region = Polygon(self.reg_pts)
    elif event == cv2.EVENT_LBUTTONUP:
        self.is_drawing = False
        self.selected_point = None
