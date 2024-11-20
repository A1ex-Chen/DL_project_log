def load_images(image_list, datadir):
    images = []
    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        im_path = os.path.join(datadir, '{}.npy'.format(frame_id))
        im = np.load(im_path)
        images.append(im)
    return np.stack(images)
