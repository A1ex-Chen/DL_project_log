def set_key(self, api_key):
    """
        Set Roboflow API key for processing.

        Args:
            api_key (str): The API key.
        """
    check_requirements('roboflow')
    from roboflow import Roboflow
    self.rf = Roboflow(api_key=api_key)
