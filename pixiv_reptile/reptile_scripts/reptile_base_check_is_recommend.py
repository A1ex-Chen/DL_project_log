def check_is_recommend(self, points):
    head = bool(points[0] > 0.85 and (points[1] > 0.85 or points[1] > 0.85))
    return head
