def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
    wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
