def make_batch_sd(image, mask, txt, device, num_samples=1):
    image = np.array(image.convert('RGB'))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    mask = np.array(mask.convert('L'))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
    masked_image = image * (mask < 0.5)
    batch = {'image': repeat(image.to(device=device), '1 ... -> n ...', n=
        num_samples), 'txt': num_samples * [txt], 'mask': repeat(mask.to(
        device=device), '1 ... -> n ...', n=num_samples), 'masked_image':
        repeat(masked_image.to(device=device), '1 ... -> n ...', n=num_samples)
        }
    return batch
