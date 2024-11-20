@staticmethod
def resize_and_crop_tensor(samples: torch.Tensor, new_width: int,
    new_height: int) ->torch.Tensor:
    orig_height, orig_width = samples.shape[2], samples.shape[3]
    if orig_height != new_height or orig_width != new_width:
        ratio = max(new_height / orig_height, new_width / orig_width)
        resized_width = int(orig_width * ratio)
        resized_height = int(orig_height * ratio)
        samples = F.interpolate(samples, size=(resized_height,
            resized_width), mode='bilinear', align_corners=False)
        start_x = (resized_width - new_width) // 2
        end_x = start_x + new_width
        start_y = (resized_height - new_height) // 2
        end_y = start_y + new_height
        samples = samples[:, :, start_y:end_y, start_x:end_x]
    return samples
