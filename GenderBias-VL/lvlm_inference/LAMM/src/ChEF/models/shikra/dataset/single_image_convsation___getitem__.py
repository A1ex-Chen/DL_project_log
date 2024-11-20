def __getitem__(self, index, debug_mode=False, return_conv=False) ->Dict[
    str, Any]:
    item = self.get_raw_item(index)
    image: Image.Image = item.get('image', None)
    target: Dict[str, Any] = item.get('target', None)
    raw_conv: List[Dict[str, Any]] = item['conversations']
    assert isinstance(image, list) == isinstance(target, list)
    multimage_mode = isinstance(image, list)
    if isinstance(image, list):
        transformed_image, transformed_target = [], []
        for img, tgt in zip(image, target):
            if self.transforms is not None and image is not None:
                img, tgt = self.transforms(img, tgt)
            if tgt is not None:
                tgt['width'], tgt['height'] = img.width, img.height
            transformed_image.append(img)
            transformed_target.append(tgt)
        image, target = transformed_image, transformed_target
    else:
        self.validate_raw_item(item)
        if self.transforms is not None and image is not None:
            image, target = self.transforms(image, target)
        has_image = 'image' in item and bool(item['image'])
        has_target = 'target' in item and bool(item['target']) and any(bool
            (elem) for elem in item['target'].values())
        if has_target and has_image:
            target['width'], target['height'] = image.width, image.height
    raw_conv = self.process_conv(raw_conv)
    raw_conv, image = self.process_conv_multimage(raw_conv, image)
    raw_conv, _ = self.process_target(raw_conv, target, multimage_mode=
        multimage_mode)
    conv = self.build_conv(raw_conv)
    if return_conv:
        return conv
    text_dict = self.process_text(conv)
    image_dict = self.process_image(image)
    ret_dict = {}
    ret_dict.update(text_dict)
    ret_dict.update(image_dict)
    self._print_sample(ret_dict, raw_conv, conv)
    if debug_mode:
        return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv,
            'image': image}
    return ret_dict
