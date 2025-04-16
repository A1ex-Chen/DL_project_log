def start_iteration(self, mode='train'):
    if mode == 'val':
        self.val_iteration += 1
    elif mode == 'train':
        self.iteration += 1
    elif mode == 'calib':
        self.calib_iteration += 1
