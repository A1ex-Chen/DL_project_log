def preprocess_images(examples):
    original_images = np.concatenate([convert_to_np(image, args.resolution) for
        image in examples[original_image_column]])
    edited_images = np.concatenate([convert_to_np(image, args.resolution) for
        image in examples[edited_image_column]])
    images = np.concatenate([original_images, edited_images])
    images = torch.tensor(images)
    images = 2 * (images / 255) - 1
    return train_transforms(images)
