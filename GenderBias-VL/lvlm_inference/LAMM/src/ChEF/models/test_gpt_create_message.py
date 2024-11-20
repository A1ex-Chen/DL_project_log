def create_message(image_list, prompts):
    msg = {'role': 'user', 'content': [{'type': 'text', 'text': prompts}]}
    img_template = {'type': 'image_url', 'image_url': {'url': '', 'detail':
        'high'}}
    url_template = 'data:image/jpeg;base64,{}'
    for image in image_list:
        img_msg = copy.deepcopy(img_template)
        base64_image = encode_image(image)
        img_msg['image_url']['url'] = url_template.format(base64_image)
        msg['content'].append(img_msg)
    return [msg]
