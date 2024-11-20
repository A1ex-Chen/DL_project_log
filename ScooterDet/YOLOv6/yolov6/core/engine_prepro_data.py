@staticmethod
def prepro_data(batch_data, device):
    images = batch_data[0].to(device, non_blocking=True).float() / 255
    targets = batch_data[1].to(device)
    return images, targets
