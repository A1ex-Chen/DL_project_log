def validate_raw_item(self, item):
    has_image = 'image' in item and bool(item['image'])
    has_target = 'target' in item and bool(item['target']) and any(bool(
        elem) for elem in item['target'].values())
    has_target_boxes = 'boxes' in item['target'] if has_target else False
    raw_conv: List[Dict[str, Any]] = item['conversations']
    human_input_has_image_placeholder = any(sentence['from'] == 'human' and
        IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv)
    if human_input_has_image_placeholder:
        assert has_image
    if has_image and not human_input_has_image_placeholder:
        warnings.warn(
            f'item has image but the question has no image placeholder.\n{item}'
            )
    gpt_input_has_image_placeholder = any(sentence['from'] == 'gpt' and 
        IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv)
    assert not gpt_input_has_image_placeholder
    has_boxes_placeholder = any(BOXES_PLACEHOLDER in sentence['value'] for
        sentence in raw_conv)
    if has_boxes_placeholder:
        assert has_target_boxes
