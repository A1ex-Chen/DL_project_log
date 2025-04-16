def __init__(self, model_path, txt_color=(0, 0, 0), bg_color=(255, 255, 255
    ), occupied_region_color=(0, 255, 0), available_region_color=(0, 0, 255
    ), margin=10):
    """
        Initializes the parking management system with a YOLOv8 model and visualization settings.

        Args:
            model_path (str): Path to the YOLOv8 model.
            txt_color (tuple): RGB color tuple for text.
            bg_color (tuple): RGB color tuple for background.
            occupied_region_color (tuple): RGB color tuple for occupied regions.
            available_region_color (tuple): RGB color tuple for available regions.
            margin (int): Margin for text display.
        """
    self.model_path = model_path
    self.model = self.load_model()
    self.labels_dict = {'Occupancy': 0, 'Available': 0}
    self.margin = margin
    self.bg_color = bg_color
    self.txt_color = txt_color
    self.occupied_region_color = occupied_region_color
    self.available_region_color = available_region_color
    self.window_name = 'Ultralytics YOLOv8 Parking Management System'
    self.env_check = check_imshow(warn=True)
