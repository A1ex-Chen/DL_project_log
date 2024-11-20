def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
    """Assign text color based on background color."""
    if color in self.dark_colors:
        return 104, 31, 17
    elif color in self.light_colors:
        return 255, 255, 255
    else:
        return txt_color
