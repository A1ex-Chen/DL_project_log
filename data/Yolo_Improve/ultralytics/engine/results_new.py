def new(self):
    """Returns a new Results object with the same image, path, names, and speed attributes."""
    return Results(orig_img=self.orig_img, path=self.path, names=self.names,
        speed=self.speed)
