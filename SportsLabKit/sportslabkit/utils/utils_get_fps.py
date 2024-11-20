def get_fps(path):
    path = str(path)
    cap = cv2.VideoCapture(path)
    return cap.get(cv2.CAP_PROP_FPS)
