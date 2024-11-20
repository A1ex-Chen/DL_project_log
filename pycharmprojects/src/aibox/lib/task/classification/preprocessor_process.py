def process(self, image: Union[PIL.Image.Image, Tensor], is_train_or_eval: bool
    ) ->Tuple[Tensor, Dict[str, Any]]:
    processed_image, process_dict = super().process(image, is_train_or_eval)
    process_dict.update({self.PROCESS_KEY_EVAL_CENTER_CROP_RATIO: self.
        eval_center_crop_ratio})
    return processed_image, process_dict
