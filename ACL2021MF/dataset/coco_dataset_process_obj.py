def process_obj(self, img_id):
    image_features, box_np, class_np = self._image_features_reader[img_id]
    if self.config.use_copy_obj:
        copy_image_features, copy_box_np, copy_class_np = (self.
            _copy_image_features_reader[img_id])
        self.process_input_for_encoder(img_id, image_features, box_np,
            class_np, copy_image_features, copy_box_np, copy_class_np)
    else:
        self.process_input_for_encoder(img_id, image_features, box_np, class_np
            )
