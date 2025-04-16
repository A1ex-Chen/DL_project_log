def process_data_dict(self, data):
    """ Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        """
    device = self.device
    img_idx = data.get('img.idx')
    camera_mat = data.get('img.camera_mat').to(device)
    scale_mat = data.get('img.scale_mat').to(device)
    return camera_mat, scale_mat, img_idx
