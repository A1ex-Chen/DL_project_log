def restore_image(image, image_info, mean=[0.485, 0.456, 0.406], std=[0.229,
    0.224, 0.225]):
    image_info = tf.cast(image_info, tf.int32)
    image = image[:image_info[0], :image_info[1]]
    image = tf.clip_by_value(image * std + mean, 0, 1) * 255
    return image
