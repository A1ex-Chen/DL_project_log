def get_images(self, return_pil=False):
    images = []
    for i, (role, msg) in enumerate(self.messages[self.offset:]):
        if i % 2 == 0:
            if type(msg) is tuple:
                import base64
                from io import BytesIO
                from PIL import Image
                msg, image, image_process_mode = msg
                if image_process_mode == 'Pad':

                    def expand2square(pil_img, background_color=(122, 116, 104)
                        ):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width),
                                background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height,
                                height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image)
                elif image_process_mode == 'Crop':
                    pass
                elif image_process_mode == 'Resize':
                    image = image.resize((224, 224))
                else:
                    raise ValueError(
                        f'Invalid image_process_mode: {image_process_mode}')
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 800, 400
                shortest_edge = int(min(max_len / aspect_ratio, min_len,
                    min_hw))
                longest_edge = int(shortest_edge * aspect_ratio)
                W, H = image.size
                if H > W:
                    H, W = longest_edge, shortest_edge
                else:
                    H, W = shortest_edge, longest_edge
                image = image.resize((W, H))
                if return_pil:
                    images.append(image)
                else:
                    buffered = BytesIO()
                    image.save(buffered, format='JPEG')
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode(
                        )
                    images.append(img_b64_str)
    return images
