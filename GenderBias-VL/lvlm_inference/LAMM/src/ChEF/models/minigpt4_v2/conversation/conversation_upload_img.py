def upload_img(self, image, conv, img_list):
    conv.append_message(conv.roles[0], '<Img><ImageHere></Img>')
    img_list.append(image)
    msg = 'Received.'
    return msg
