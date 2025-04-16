def crop_image(input_image):
    crop_w, crop_h = np.floor(np.array(input_image.size) / 64).astype(int) * 64
    im_cropped = Image.fromarray(np.array(input_image)[:crop_h, :crop_w])
    return im_cropped
