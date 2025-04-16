@staticmethod
def calculate_centroid(box):
    """
        Calculates the centroid of a bounding box.

        Args:
            box (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            (tuple): Centroid coordinates (x, y).
        """
    return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)
