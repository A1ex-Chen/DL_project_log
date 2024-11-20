def checkext(self, path):
    if self.webcam:
        file_type = 'video'
    else:
        file_type = 'image' if path.split('.')[-1].lower(
            ) in IMG_FORMATS else 'video'
    return file_type
