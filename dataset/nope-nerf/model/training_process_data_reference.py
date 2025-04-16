def process_data_reference(self, data):
    """ Processes the data dictionary and returns respective tensors
        Args:
            data (dictionary): data dictionary
        """
    device = self.device
    ref_imgs = data.get('img.ref_imgs').to(device)
    ref_dpts = data.get('img.ref_dpts').to(device).unsqueeze(1)
    ref_idxs = data.get('img.ref_idxs')
    return ref_imgs, ref_dpts, ref_idxs
