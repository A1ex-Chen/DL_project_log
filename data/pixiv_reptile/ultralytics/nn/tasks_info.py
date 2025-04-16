def info(self, detailed=False, verbose=True, imgsz=640):
    """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
    return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
