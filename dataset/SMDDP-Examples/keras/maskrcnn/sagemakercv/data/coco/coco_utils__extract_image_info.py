def _extract_image_info(prediction, b):
    return {'id': int(prediction['source_id'][b]), 'width': int(prediction[
        'width'][b]), 'height': int(prediction['height'][b])}
