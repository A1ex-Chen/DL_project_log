def process_data(self, json_data, im0, boxes, clss):
    """
        Process the model data for parking lot management.

        Args:
            json_data (str): json data for parking lot management
            im0 (ndarray): inference image
            boxes (list): bounding boxes data
            clss (list): bounding boxes classes list

        Returns:
            filled_slots (int): total slots that are filled in parking lot
            empty_slots (int): total slots that are available in parking lot
        """
    annotator = Annotator(im0)
    total_slots, filled_slots = len(json_data), 0
    empty_slots = total_slots
    for region in json_data:
        points = region['points']
        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        region_occupied = False
        for box, cls in zip(boxes, clss):
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
            text = f'{self.model.names[int(cls)]}'
            annotator.display_objects_labels(im0, text, self.txt_color,
                self.bg_color, x_center, y_center, self.margin)
            dist = cv2.pointPolygonTest(points_array, (x_center, y_center),
                False)
            if dist >= 0:
                region_occupied = True
                break
        color = (self.occupied_region_color if region_occupied else self.
            available_region_color)
        cv2.polylines(im0, [points_array], isClosed=True, color=color,
            thickness=2)
        if region_occupied:
            filled_slots += 1
            empty_slots -= 1
    self.labels_dict['Occupancy'] = filled_slots
    self.labels_dict['Available'] = empty_slots
    annotator.display_analytics(im0, self.labels_dict, self.txt_color, self
        .bg_color, self.margin)
