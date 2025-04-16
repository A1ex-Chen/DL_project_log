def transform_images(examples):
    images = [augmentations(image.convert('RGB')) for image in examples[
        args.dataset_image_column_name]]
    batch_dict = {'images': images}
    if args.class_conditional:
        batch_dict['class_labels'] = examples[args.
            dataset_class_label_column_name]
    return batch_dict
