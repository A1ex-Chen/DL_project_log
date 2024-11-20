def prepare_image(pil_image):
    arr = np.array(pil_image.convert('RGB'))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    return image
