def train_transforms(img):
    img = center_crop(img)
    img = img.resize((args.resolution, args.resolution), resample=Image.
        BICUBIC, reducing_gap=1)
    img = np.array(img).astype(np.float32) / 127.5 - 1
    img = torch.from_numpy(np.transpose(img, [2, 0, 1]))
    return img
