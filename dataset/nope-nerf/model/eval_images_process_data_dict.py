def process_data_dict(self, data):
    """ Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        """
    device = self.device
    img = data.get('img').to(device)
    batch_size, _, h, w = img.shape
    depth_img = data.get('img.depth', torch.ones(batch_size, h, w))
    img_idx = data.get('img.idx')
    camera_mat = data.get('img.camera_mat').to(device)
    scale_mat = data.get('img.scale_mat').to(device)
    return img, depth_img, camera_mat, scale_mat, img_idx
